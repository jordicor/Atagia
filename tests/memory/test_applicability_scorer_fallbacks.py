"""Tests for applicability scoring fallbacks and superseded/stale demotion."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings, default_resource_path
from atagia.memory.applicability_scorer import ApplicabilityScorer
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver
from atagia.models.schemas_memory import (
    ExtractionContextMessage,
    ExtractionConversationContext,
    RetrievalPlan,
)
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)

MANIFESTS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "manifests"
)


class CannedCardProvider(LLMProvider):
    name = "canned-card-fallbacks"

    def __init__(self, *, relevance_output: str) -> None:
        self.relevance_output = relevance_output
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        purpose = str(request.metadata.get("purpose") or "")
        assert purpose == "applicability_relevance_card"
        output_text = self.relevance_output
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=output_text,
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embeddings are not used by applicability scorer tests")


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


def _scorer(provider: LLMProvider) -> ApplicabilityScorer:
    return ApplicabilityScorer(
        llm_client=LLMClient(
            provider_name="canned-card-fallbacks", providers=[provider]
        ),
        clock=FrozenClock(datetime(2026, 3, 30, 21, 0, tzinfo=timezone.utc)),
        settings=_settings(),
    )


def _resolved_policy(mode_id: str = "coding_debug"):
    loader = ManifestLoader(MANIFESTS_DIR)
    manifest = loader.load_all()[mode_id]
    return PolicyResolver().resolve(manifest, None, None)


def _context() -> ExtractionConversationContext:
    return ExtractionConversationContext(
        user_id="usr_1",
        conversation_id="cnv_1",
        source_message_id="msg_1",
        workspace_id=None,
        assistant_mode_id="coding_debug",
        recent_messages=[
            ExtractionContextMessage(
                role="user", content="It still failed in production."
            ),
        ],
    )


def _plan() -> RetrievalPlan:
    fts_queries = ["safe outage fix"]
    return RetrievalPlan(
        assistant_mode_id="coding_debug",
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
        max_candidates=10,
        max_context_items=8,
        privacy_ceiling=1,
        retrieval_levels=[0],
    )


def _candidate(
    memory_id: str,
    *,
    status: str = "active",
    valid_from: str | None = None,
    valid_to: str | None = None,
    temporal_type: str = "unknown",
    rrf_score: float | None = 0.5,
    object_type: str = "evidence",
) -> dict[str, object]:
    candidate: dict[str, object] = {
        "id": memory_id,
        "user_id": "usr_1",
        "conversation_id": "cnv_1",
        "assistant_mode_id": "coding_debug",
        "platform_id": "default",
        "platform_locked": 0,
        "platform_id_lock": None,
        "object_type": object_type,
        "scope": "conversation",
        "scope_canonical": "chat",
        "canonical_text": "The websocket retry loop still fails in production.",
        "payload_json": {},
        "privacy_level": 0,
        "sensitivity": "public",
        "temporal_type": temporal_type,
        "valid_from": valid_from,
        "valid_to": valid_to,
        "status": status,
        "updated_at": "2026-03-30T21:00:00+00:00",
        "rank": 0.5,
        "vitality": 0.0,
        "maya_score": 0.0,
        "retrieval_sources": ["fts"],
    }
    if rrf_score is not None:
        candidate["rrf_score"] = rrf_score
    return candidate


@pytest.mark.asyncio
async def test_unparsed_llm_score_falls_back_to_retrieval_score() -> None:
    provider = CannedCardProvider(
        relevance_output="not a parseable card",
    )
    scorer = _scorer(provider)

    scored = await scorer.score_shortlist(
        [_candidate("mem_fallback", rrf_score=0.6)],
        message_text="What context matters here?",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        detected_needs=[],
        retrieval_plan=_plan(),
    )

    # The candidate is kept: its rrf score is mapped into the sub-LLM
    # applicability range [0.05, 0.55] instead of being dropped, so it can
    # never outrank an honestly scored "useful" candidate.
    assert [item.memory_id for item in scored] == ["mem_fallback"]
    assert scored[0].llm_applicability == pytest.approx(0.05 + 0.6 * 0.50)


@pytest.mark.asyncio
async def test_unparsed_llm_score_with_missing_rrf_uses_neutral_midpoint() -> None:
    provider = CannedCardProvider(
        relevance_output="not a parseable card",
    )
    scorer = _scorer(provider)

    scored = await scorer.score_shortlist(
        [_candidate("mem_no_rrf", rrf_score=None)],
        message_text="What context matters here?",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        detected_needs=[],
        retrieval_plan=_plan(),
    )

    assert [item.memory_id for item in scored] == ["mem_no_rrf"]
    assert scored[0].llm_applicability == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_superseded_candidate_ranks_below_current_but_stays_in_pool() -> None:
    provider = CannedCardProvider(
        relevance_output="candidate_000 useful\ncandidate_001 useful",
    )
    scorer = _scorer(provider)

    scored = await scorer.score_shortlist(
        [
            _candidate("mem_current", rrf_score=0.6),
            _candidate("mem_superseded", status="superseded", rrf_score=0.6),
        ],
        message_text="What context matters here?",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        detected_needs=[],
        retrieval_plan=_plan(),
    )

    # Demoted, never dropped: both survive and the superseded memory ranks
    # exactly one penalty step below the equivalent current one.
    assert [item.memory_id for item in scored] == ["mem_current", "mem_superseded"]
    by_id = {item.memory_id: item for item in scored}
    assert by_id["mem_superseded"].penalty == pytest.approx(
        by_id["mem_current"].penalty + 0.05
    )
    assert by_id["mem_current"].final_score - by_id[
        "mem_superseded"
    ].final_score == pytest.approx(0.05)


@pytest.mark.asyncio
async def test_expired_valid_window_is_demoted_but_not_dropped() -> None:
    provider = CannedCardProvider(
        relevance_output="candidate_000 useful\ncandidate_001 useful",
    )
    scorer = _scorer(provider)

    scored = await scorer.score_shortlist(
        [
            _candidate(
                "mem_expired",
                temporal_type="bounded",
                valid_from="2026-02-01T00:00:00+00:00",
                valid_to="2026-03-01T00:00:00+00:00",
                rrf_score=0.6,
            ),
            _candidate(
                "mem_still_valid",
                temporal_type="bounded",
                valid_from="2026-02-01T00:00:00+00:00",
                valid_to="2026-12-01T00:00:00+00:00",
                rrf_score=0.6,
            ),
        ],
        message_text="What context matters here?",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        detected_needs=[],
        retrieval_plan=_plan(),
    )

    assert [item.memory_id for item in scored] == ["mem_still_valid", "mem_expired"]
    by_id = {item.memory_id: item for item in scored}
    assert by_id["mem_expired"].penalty == pytest.approx(0.05)
    assert by_id["mem_still_valid"].penalty == pytest.approx(0.0)


def test_ephemeral_candidates_do_not_get_double_stale_penalty() -> None:
    scorer = _scorer(CannedCardProvider(relevance_output="candidate_000 useful"))

    penalty = scorer._penalty(
        _candidate(
            "mem_ephemeral",
            temporal_type="ephemeral",
            valid_from="2026-03-29T10:00:00+00:00",
        )
    )

    # Stale ephemerals keep the existing single 0.08 penalty; the new
    # expired-window penalty must not stack on top.
    assert penalty == pytest.approx(0.08)


@pytest.mark.asyncio
async def test_summary_view_is_demoted_below_direct_evidence_but_kept() -> None:
    provider = CannedCardProvider(
        relevance_output="candidate_000 useful\ncandidate_001 useful",
    )
    scorer = _scorer(provider)

    scored = await scorer.score_shortlist(
        [
            _candidate("mem_summary", object_type="summary_view", rrf_score=0.6),
            _candidate("mem_direct", rrf_score=0.6),
        ],
        message_text="What context matters here?",
        conversation_context=_context(),
        resolved_policy=_resolved_policy(),
        detected_needs=[],
        retrieval_plan=_plan(),
    )

    # Summaries are retrieval aids, not canonical truth: the direct-evidence
    # candidate outranks the equally scored summary, but the summary stays
    # in the pool for episode-level questions.
    assert [item.memory_id for item in scored] == ["mem_direct", "mem_summary"]
    by_id = {item.memory_id: item for item in scored}
    assert by_id["mem_summary"].penalty == pytest.approx(
        by_id["mem_direct"].penalty + 0.05
    )
