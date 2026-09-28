"""Ablation handling on the fast/smart_fast context path.

The fast path never enters the retrieval pipeline, so it can neither apply a
retrieval override nor produce a differently-composed context. It used to
ignore both facts: it served a warm cache entry to a turn that had disabled the
cache, dropped every override without a word, and returned the cache fields of
a turn that had computed everything itself. These tests pin the three
behaviors that replaced that -- refuse the read the normal path would refuse,
refuse the request the path cannot honor, and report the cache state actually
used.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atagia.app import AppRuntime
from atagia.models.schemas_memory import ResponseMode
from atagia.models.schemas_replay import AblationConfig
from atagia.services.context_cache_service import (
    ContextCacheService,
    FastModeAblationUnsupportedError,
)
from atagia.services.prompt_authority import normalize_request_authority_context

from tests.services.test_adaptive_retrieval_gate_cache import (
    _build_runtime,
    _seed,
)

_QUESTION = "Do I prefer concise replies?"


def _authority() -> object:
    return normalize_request_authority_context(
        privacy_enforcement="enforce",
        authenticated_user_privilege_level=None,
        authenticated_user_is_atagia_master=False,
        user_id="usr_1",
        purpose="context_cache_fast",
    )


async def _warm_smart_fast_entry(runtime: AppRuntime) -> ContextCacheService:
    """Publish one smart_fast warm entry the fast path can later read."""
    service = ContextCacheService(runtime)
    connection = await runtime.open_connection()
    try:
        warm = await service.resolve_with_connection(
            connection,
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text=_QUESTION,
            response_mode=ResponseMode.SMART_FAST,
            prompt_authority_context=_authority(),
        )
    finally:
        await connection.close()
    assert warm.pending_cache_entry is not None
    assert await service.publish_pending_cache_entry(
        warm, last_retrieval_message_seq=1
    )
    assert warm.composed_context.selected_memory_ids == ["mem_1"]
    return service


async def _resolve_fast(
    runtime: AppRuntime,
    service: ContextCacheService,
    *,
    ablation: AblationConfig | None,
):
    connection = await runtime.open_connection()
    try:
        return await service.resolve_fast_with_connection(
            connection,
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text=_QUESTION,
            response_mode=ResponseMode.SMART_FAST,
            prompt_authority_context=_authority(),
            ablation=ablation,
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_smart_fast_serves_the_warm_entry_and_labels_it_as_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A served warm entry is reported as a cache hit, not as a fresh turn."""
    runtime, _provider = await _build_runtime(
        tmp_path, monkeypatch, memory_dependence="mixed"
    )
    try:
        await _seed(runtime)
        service = await _warm_smart_fast_entry(runtime)

        fast = await _resolve_fast(runtime, service, ablation=None)

        assert fast.source_retrieval_plan["smart_fast_warm_read_allowed"] is True
        assert fast.source_retrieval_plan["smart_fast_warm_entry_present"] is True
        assert fast.composed_context.selected_memory_ids == ["mem_1"]
        assert fast.from_cache is True
        assert fast.cache_source == "cache_hit"
        assert fast.next_refresh_strategy == "cache"
        assert isinstance(fast.cache_age_seconds, float)
        assert fast.cache_age_seconds >= 0.0
        assert fast.retrieval_custody_v2_status == "cache_hit_no_candidate_custody"
        assert fast.sufficiency_diagnostics_v1_status == (
            "cache_hit_no_sufficiency_diagnostics"
        )
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_smart_fast_refuses_the_warm_entry_when_the_cache_is_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``disable_context_cache`` has to hold on the path that can serve it.

    The normal path refuses the read for this ablation. The fast path did not
    even ask, so the caller that switched the cache off still got the cached
    memory context -- and every cache field said the turn had used no cache.
    """
    runtime, _provider = await _build_runtime(
        tmp_path, monkeypatch, memory_dependence="mixed"
    )
    try:
        await _seed(runtime)
        service = await _warm_smart_fast_entry(runtime)

        fast = await _resolve_fast(
            runtime,
            service,
            ablation=AblationConfig(disable_context_cache=True),
        )

        assert fast.source_retrieval_plan["smart_fast_warm_read_allowed"] is False
        assert fast.source_retrieval_plan["smart_fast_warm_entry_present"] is False
        assert fast.composed_context.selected_memory_ids == []
        assert fast.memory_summaries == []
        assert fast.from_cache is False
        assert fast.cache_source is None
        assert fast.cache_age_seconds is None
        assert fast.next_refresh_strategy == "sync"
        # No cache was read, so the reason there is no custody cannot be a
        # cache hit.
        assert fast.retrieval_custody_v2_status == "fast_mode_no_candidate_custody"
        assert fast.sufficiency_diagnostics_v1_status == (
            "fast_mode_no_sufficiency_diagnostics"
        )
        # Refusing to READ is not a delete: the entry stays for the turns that
        # are entitled to it.
        stored = await runtime.storage_backend.get_context_view(str(fast.cache_key))
        assert stored is not None
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_smart_fast_refuses_the_warm_entry_under_an_envelope_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An envelope the caller moved is not the envelope the entry was built for."""
    runtime, _provider = await _build_runtime(
        tmp_path, monkeypatch, memory_dependence="mixed"
    )
    try:
        await _seed(runtime)
        service = await _warm_smart_fast_entry(runtime)

        fast = await _resolve_fast(
            runtime,
            service,
            ablation=AblationConfig(context_envelope_budget_tokens=1_000),
        )

        assert fast.source_retrieval_plan["smart_fast_warm_read_allowed"] is False
        assert fast.composed_context.selected_memory_ids == []
        assert fast.from_cache is False
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_smart_fast_refuses_a_transcript_budget_override_read_but_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one override the caller applies is accepted, and recorded as applied.

    ``transcript_budget_tokens`` is consumed by the calling surface, so the turn
    runs; the warm entry is still refused because it was composed without that
    override, exactly as the normal path refuses it.
    """
    runtime, _provider = await _build_runtime(
        tmp_path, monkeypatch, memory_dependence="mixed"
    )
    try:
        await _seed(runtime)
        service = await _warm_smart_fast_entry(runtime)

        fast = await _resolve_fast(
            runtime,
            service,
            ablation=AblationConfig(
                override_retrieval_params={"transcript_budget_tokens": 700}
            ),
        )

        assert fast.source_retrieval_plan["smart_fast_warm_read_allowed"] is False
        assert fast.from_cache is False
        assert fast.retrieval_trace is not None
        assert fast.retrieval_trace["applied_override_retrieval_params"] == {
            "transcript_budget_tokens": 700
        }
    finally:
        await runtime.close()


@pytest.mark.parametrize(
    ("ablation", "expected_fragment"),
    [
        (
            AblationConfig(override_retrieval_params={"context_budget_tokens": 500}),
            "context_budget_tokens",
        ),
        (
            AblationConfig(override_retrieval_params={"privacy_ceiling": 3}),
            "privacy_ceiling",
        ),
        (
            AblationConfig(composer_strategy="budgeted_marginal"),
            "composer_strategy",
        ),
        (
            AblationConfig(applicability_gate_mode="enforced"),
            "applicability_gate_mode",
        ),
        (AblationConfig(force_all_scopes=True), "force_all_scopes"),
        (
            AblationConfig(enable_llm_coverage_expansion=True),
            "enable_llm_coverage_expansion",
        ),
        (
            AblationConfig(enable_final_answer_evidence_pack=True),
            "enable_final_answer_evidence_pack",
        ),
    ],
)
@pytest.mark.asyncio
async def test_smart_fast_fails_fast_on_an_ablation_it_cannot_honor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ablation: AblationConfig,
    expected_fragment: str,
) -> None:
    """Silently dropping the request is what produced dishonest runs."""
    runtime, _provider = await _build_runtime(
        tmp_path, monkeypatch, memory_dependence="mixed"
    )
    try:
        await _seed(runtime)
        service = ContextCacheService(runtime)

        with pytest.raises(FastModeAblationUnsupportedError) as excinfo:
            await _resolve_fast(runtime, service, ablation=ablation)

        assert expected_fragment in str(excinfo.value)
    finally:
        await runtime.close()


@pytest.mark.parametrize(
    "ablation",
    [
        AblationConfig(privacy_enforcement="off"),
        AblationConfig(skip_contract_memory=True),
        AblationConfig(skip_need_detection=True),
        AblationConfig(skip_applicability_scoring=True),
        AblationConfig(skip_fusion_dedupe=True),
        AblationConfig(skip_workspace_rollup=True),
        AblationConfig(applicability_gate_mode="off"),
        AblationConfig(enable_evidence_packets=False),
        AblationConfig(enable_evidence_obligation_coverage=False),
    ],
)
@pytest.mark.asyncio
async def test_smart_fast_accepts_an_ablation_it_honors_or_already_satisfies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ablation: AblationConfig,
) -> None:
    """Refusing a knob the mode already satisfies would reject valid requests."""
    runtime, _provider = await _build_runtime(
        tmp_path, monkeypatch, memory_dependence="mixed"
    )
    try:
        await _seed(runtime)
        service = ContextCacheService(runtime)

        fast = await _resolve_fast(runtime, service, ablation=ablation)

        assert fast.source_retrieval_plan["fast_mode"] is True
    finally:
        await runtime.close()
