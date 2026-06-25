"""Concurrency-overlap tests for the retrieval pipeline (F2.1).

These cover the two LLM/SQL overlaps introduced in ``execute``:

* Overlap A: the need-detection LLM call runs concurrently with the base
  candidate-search SQL (after the profiles, which feed the need prompt, have
  completed sequentially).
* Overlap B: the applicability-scoring LLM call runs concurrently with the
  bundled contract/state/workspace-rollup SQL lookups.

The proofs are deterministic: each side records start/end timestamps around a
generous sleep and we assert the intervals genuinely interleave, rather than
relying on a fragile elapsed-time threshold. Failure-path tests assert the
existing fallback/fail-fast semantics are preserved and that the overlap task
is never left orphaned.
"""

from __future__ import annotations

import asyncio
import warnings

import pytest

from atagia.models.schemas_memory import MemoryScope, RetrievalTrace
from atagia.services.llm_client import (
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMError,
)

from tests.services.test_retrieval_pipeline import (
    FailingPipelineProvider,
    PipelineProvider,
    _build_runtime,
    _seed_memory,
)

# The LLM side sleeps longer than the SQL side so the SQL coroutine
# deterministically finishes *inside* the LLM window, proving genuine overlap
# (not merely adjacent windows) without relying on tight wall-clock thresholds.
_LLM_SLEEP_S = 0.20
_SQL_SLEEP_S = 0.05


def _intervals_overlap(
    a_start: float,
    a_end: float,
    b_start: float,
    b_end: float,
) -> bool:
    """Return True when two [start, end] intervals genuinely interleave."""
    return a_start < b_end and b_start < a_end


class _TimedLLMProvider(PipelineProvider):
    """Provider that times a chosen purpose around a generous sleep.

    The sleep widens the LLM window so the partnered SQL coroutine has room to
    run inside it, making the interleave deterministic without wall-clock
    threshold guesswork.
    """

    def __init__(self, timed_purpose: str, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._timed_purpose = timed_purpose
        self.timed_start: float | None = None
        self.timed_end: float | None = None

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        purpose = str(request.metadata.get("purpose"))
        timed_match = purpose == self._timed_purpose or (
            self._timed_purpose == "need_detection"
            and purpose.startswith("need_detection_")
            and purpose.endswith("_card")
        ) or (
            self._timed_purpose == "applicability_scoring"
            and purpose == "applicability_relevance_card"
        )
        if timed_match:
            self.timed_start = asyncio.get_event_loop().time()
            await asyncio.sleep(_LLM_SLEEP_S)
            response = await super().complete(request)
            self.timed_end = asyncio.get_event_loop().time()
            return response
        return await super().complete(request)


def _wrap_base_search_timing(pipeline: object) -> dict[str, float | None]:
    """Wrap the base candidate-search call to record its [start, end].

    Only the *base* search is timed: the base call passes no
    ``runtime_alias_groups`` kwarg, while the enriched call always does.
    """
    timings: dict[str, float | None] = {"start": None, "end": None}
    search = pipeline._candidate_search  # type: ignore[attr-defined]
    original = search.search

    async def _timed(plan, user_id, *args, **kwargs):  # type: ignore[no-untyped-def]
        if "runtime_alias_groups" not in kwargs:
            timings["start"] = asyncio.get_event_loop().time()
            await asyncio.sleep(_SQL_SLEEP_S)
            result = await original(plan, user_id, *args, **kwargs)
            timings["end"] = asyncio.get_event_loop().time()
            return result
        return await original(plan, user_id, *args, **kwargs)

    search.search = _timed  # type: ignore[attr-defined]
    return timings


def _wrap_lookups_timing(pipeline: object) -> dict[str, float | None]:
    """Wrap ``_post_scoring_lookups`` to record its [start, end]."""
    timings: dict[str, float | None] = {"start": None, "end": None}
    original = pipeline._post_scoring_lookups  # type: ignore[attr-defined]

    async def _timed(**kwargs):  # type: ignore[no-untyped-def]
        timings["start"] = asyncio.get_event_loop().time()
        await asyncio.sleep(_SQL_SLEEP_S)
        result = await original(**kwargs)
        timings["end"] = asyncio.get_event_loop().time()
        return result

    pipeline._post_scoring_lookups = _timed  # type: ignore[attr-defined]
    return timings


def _full_flow_provider() -> PipelineProvider:
    return PipelineProvider(
        need_response={
            "needs": [],
            "temporal_range": None,
            "sub_queries": ["retry loop websocket backoff"],
            "sparse_query_hints": [
                {
                    "sub_query_text": "retry loop websocket backoff",
                    "fts_phrase": "retry loop websocket backoff",
                }
            ],
            "query_type": "default",
            "retrieval_levels": [0],
        },
        score_map={"mem_1": 0.91},
    )


async def _seed_full_flow_corpus(memories: object) -> None:
    await _seed_memory(
        memories,
        memory_id="mem_1",
        canonical_text="retry loop websocket backoff",
        scope=MemoryScope.CONVERSATION,
    )
    await _seed_memory(
        memories,
        memory_id="mem_user",
        canonical_text="User prefers concise debugging answers.",
        scope=MemoryScope.GLOBAL_USER,
    )


# --------------------------------------------------------------------------- #
# Overlap A: need detection (LLM) ‖ base candidate search (SQL)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_overlap_a_need_detection_runs_concurrently_with_base_search() -> None:
    provider = _TimedLLMProvider(
        "need_detection",
        need_response=_full_flow_provider().need_response,
        score_map={"mem_1": 0.91},
    )
    connection, memories, _contracts, pipeline, _provider, policy, context = (
        await _build_runtime(provider=provider)
    )
    base_timings = _wrap_base_search_timing(pipeline)
    try:
        await _seed_full_flow_corpus(memories)
        result = await pipeline.execute(
            message_text="retry loop websocket backoff",
            conversation_context=context,
            resolved_policy=policy,
            cold_start=False,
            conversation_messages=[
                {"role": "user", "text": "retry loop websocket backoff"}
            ],
        )

        assert provider.timed_start is not None and provider.timed_end is not None
        assert base_timings["start"] is not None and base_timings["end"] is not None
        # The two windows genuinely interleave: base SQL ran while the
        # need-detection LLM call was still in flight.
        assert _intervals_overlap(
            provider.timed_start,
            provider.timed_end,
            base_timings["start"],
            base_timings["end"],
        )
        # Stronger claim: base search completed while the need-detection LLM
        # call was still in flight (the SQL window closed before the LLM did).
        assert base_timings["end"] < provider.timed_end
        # Both stages still recorded their own (non-aggregated) durations.
        assert result.stage_timings["base_candidate_search"] > 0.0
        assert result.stage_timings["need_detection"] > 0.0
        # Pipeline still produces the expected result.
        assert result.degraded_mode is False
        assert [candidate["id"] for candidate in result.raw_candidates] == ["mem_1"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_overlap_a_logs_stage_overlap_annotation(caplog) -> None:
    provider = _full_flow_provider()
    connection, memories, _contracts, pipeline, _provider, policy, context = (
        await _build_runtime(provider=provider)
    )
    try:
        await _seed_full_flow_corpus(memories)
        with caplog.at_level("INFO", logger="atagia.services.retrieval_pipeline"):
            await pipeline.execute(
                message_text="retry loop websocket backoff",
                conversation_context=context,
                resolved_policy=policy,
                cold_start=False,
                conversation_messages=[
                    {"role": "user", "text": "retry loop websocket backoff"}
                ],
            )
        overlap_records = [
            record
            for record in caplog.records
            if record.message == "retrieval_stage_overlap"
        ]
        annotated = {
            stage: partner
            for record in overlap_records
            for stage, partner in getattr(record, "stage_overlaps", {}).items()
        }
        assert annotated["base_candidate_search"] == "need_detection"
        assert annotated["contract_lookup"] == "applicability_scoring"
        assert annotated["state_lookup"] == "applicability_scoring"
        assert annotated["workspace_rollup_lookup"] == "applicability_scoring"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_overlap_a_need_detection_failure_uses_base_results() -> None:
    """Need detection raising during the overlap engages the degraded fallback
    and the base-search results still drive retrieval."""
    provider = FailingPipelineProvider(
        "need_detection",
        score_map={"mem_1": 0.91},
    )
    connection, memories, _contracts, pipeline, _provider, policy, context = (
        await _build_runtime(provider=provider)
    )
    try:
        await _seed_full_flow_corpus(memories)
        result = await pipeline.execute(
            message_text="retry loop websocket backoff",
            conversation_context=context,
            resolved_policy=policy,
            cold_start=False,
            conversation_messages=[
                {"role": "user", "text": "retry loop websocket backoff"}
            ],
        )

        assert result.degraded_mode is True
        assert result.detected_needs == []
        # Base search still ran and its candidate feeds the fallback path; the
        # scorer still runs over the base candidates.
        assert [candidate["id"] for candidate in result.raw_candidates] == ["mem_1"]
        assert [scored.memory_id for scored in result.scored_candidates] == ["mem_1"]
        assert result.composed_context.selected_memory_ids == ["mem_1"]
        # The base search timing is still recorded under the overlap reorder.
        assert result.stage_timings["base_candidate_search"] > 0.0
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_overlap_a_base_search_failure_raises_without_orphan_task() -> None:
    provider = _full_flow_provider()
    connection, memories, _contracts, pipeline, _provider, policy, context = (
        await _build_runtime(provider=provider)
    )
    search = pipeline._candidate_search
    original = search.search

    async def _failing(plan, user_id, *args, **kwargs):  # type: ignore[no-untyped-def]
        if "runtime_alias_groups" not in kwargs:
            # Yield once so the need-detection LLM call is genuinely in flight
            # when the base SQL task raises.
            await asyncio.sleep(0)
            raise RuntimeError("injected base search failure")
        return await original(plan, user_id, *args, **kwargs)

    search.search = _failing
    try:
        await _seed_full_flow_corpus(memories)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ResourceWarning)
            with pytest.raises(RuntimeError, match="injected base search failure"):
                await pipeline.execute(
                    message_text="retry loop websocket backoff",
                    conversation_context=context,
                    resolved_policy=policy,
                    cold_start=False,
                    conversation_messages=[
                        {"role": "user", "text": "retry loop websocket backoff"}
                    ],
                )
        # Let the loop process any lingering callbacks; a destroyed-pending
        # task would surface here.
        await asyncio.sleep(0)
        pending = [
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task() and not task.done()
        ]
        assert pending == []
    finally:
        await connection.close()


# --------------------------------------------------------------------------- #
# Overlap B: applicability scoring (LLM) ‖ contract/state/rollup lookups (SQL)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_overlap_b_scoring_runs_concurrently_with_lookups() -> None:
    provider = _TimedLLMProvider(
        "applicability_scoring",
        need_response=_full_flow_provider().need_response,
        score_map={"mem_1": 0.91},
    )
    connection, memories, _contracts, pipeline, _provider, policy, context = (
        await _build_runtime(provider=provider)
    )
    lookup_timings = _wrap_lookups_timing(pipeline)
    try:
        await _seed_full_flow_corpus(memories)
        result = await pipeline.execute(
            message_text="retry loop websocket backoff",
            conversation_context=context,
            resolved_policy=policy,
            cold_start=False,
            conversation_messages=[
                {"role": "user", "text": "retry loop websocket backoff"}
            ],
        )

        assert provider.timed_start is not None and provider.timed_end is not None
        assert lookup_timings["start"] is not None and lookup_timings["end"] is not None
        assert _intervals_overlap(
            provider.timed_start,
            provider.timed_end,
            lookup_timings["start"],
            lookup_timings["end"],
        )
        # Stronger claim: the lookups completed while the scoring LLM call was
        # still in flight (the SQL window closed before the LLM window did).
        assert lookup_timings["end"] < provider.timed_end
        # Each lookup stage still recorded its own duration.
        assert result.stage_timings["contract_lookup"] >= 0.0
        assert result.stage_timings["state_lookup"] >= 0.0
        assert result.stage_timings["workspace_rollup_lookup"] >= 0.0
        assert result.stage_timings["applicability_scoring"] > 0.0
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_overlap_b_lookups_failure_raises_without_orphan_task() -> None:
    provider = _full_flow_provider()
    connection, memories, _contracts, pipeline, _provider, policy, context = (
        await _build_runtime(provider=provider)
    )
    original = pipeline._post_scoring_lookups

    async def _failing(**kwargs):  # type: ignore[no-untyped-def]
        await asyncio.sleep(0)
        raise RuntimeError("injected lookups failure")

    pipeline._post_scoring_lookups = _failing
    try:
        await _seed_full_flow_corpus(memories)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ResourceWarning)
            with pytest.raises(RuntimeError, match="injected lookups failure"):
                await pipeline.execute(
                    message_text="retry loop websocket backoff",
                    conversation_context=context,
                    resolved_policy=policy,
                    cold_start=False,
                    conversation_messages=[
                        {"role": "user", "text": "retry loop websocket backoff"}
                    ],
                )
        await asyncio.sleep(0)
        pending = [
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task() and not task.done()
        ]
        assert pending == []
    finally:
        # Restore so connection.close() teardown is unaffected.
        pipeline._post_scoring_lookups = original
        await connection.close()


@pytest.mark.asyncio
async def test_overlap_b_scoring_failure_drains_lookups_task() -> None:
    """When applicability scoring fails, the error surfaces and the hoisted
    lookups task is not left orphaned."""
    provider = FailingPipelineProvider(
        "applicability_scoring",
        need_response=_full_flow_provider().need_response,
        score_map={"mem_1": 0.91},
    )
    connection, memories, _contracts, pipeline, _provider, policy, context = (
        await _build_runtime(provider=provider)
    )
    try:
        await _seed_full_flow_corpus(memories)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ResourceWarning)
            with pytest.raises(LLMError, match="Injected applicability_scoring failure"):
                await pipeline.execute(
                    message_text="retry loop websocket backoff",
                    conversation_context=context,
                    resolved_policy=policy,
                    cold_start=False,
                    conversation_messages=[
                        {"role": "user", "text": "retry loop websocket backoff"}
                    ],
                )
        await asyncio.sleep(0)
        pending = [
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task() and not task.done()
        ]
        assert pending == []
    finally:
        await connection.close()


# --------------------------------------------------------------------------- #
# Results parity: the reorder must not change the composed output.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_overlap_preserves_composed_output_parity() -> None:
    """A full pipeline run with canned fakes produces the expected composed
    context and contract/state lookups after the reorder."""
    provider = _full_flow_provider()
    connection, memories, contracts, pipeline, _provider, policy, context = (
        await _build_runtime(provider=provider)
    )
    try:
        await _seed_full_flow_corpus(memories)
        trace = RetrievalTrace(
            query_text="retry loop websocket backoff",
            user_id="usr_1",
            conversation_id="cnv_1",
            timestamp_iso="2026-04-09T12:00:00Z",
        )
        result = await pipeline.execute(
            message_text="retry loop websocket backoff",
            conversation_context=context,
            resolved_policy=policy,
            cold_start=False,
            conversation_messages=[
                {"role": "user", "text": "retry loop websocket backoff"}
            ],
            trace=trace,
        )

        # Candidate set, scoring, and composition are stable.
        assert [candidate["id"] for candidate in result.raw_candidates] == ["mem_1"]
        assert [c.memory_id for c in result.scored_candidates] == ["mem_1"]
        assert result.composed_context.selected_memory_ids == ["mem_1"]
        # The composed memory block carries the selected memory's text, so the
        # reorder provably did not alter what reaches the prompt.
        assert "retry loop websocket backoff" in result.composed_context.memory_block
        assert result.degraded_mode is False
        assert result.small_corpus_mode is False
        # Overlap B lookups still populate their result fields.
        assert isinstance(result.current_contract, dict)
        assert isinstance(result.user_state, dict)
        # All overlapped stages are present in the timing map and non-negative.
        assert {
            "need_detection",
            "base_candidate_search",
            "candidate_search",
            "applicability_scoring",
            "contract_lookup",
            "state_lookup",
            "workspace_rollup_lookup",
            "context_composition",
        }.issubset(result.stage_timings)
        assert all(value >= 0.0 for value in result.stage_timings.values())
    finally:
        await connection.close()
