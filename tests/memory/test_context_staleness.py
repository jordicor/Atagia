"""Tests for deterministic context-cache staleness scoring."""

from __future__ import annotations

from datetime import datetime, timezone
import html
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.memory.context_staleness import ContextStalenessScorer
from atagia.memory.policy_manifest import (
    ManifestLoader,
    PolicyResolver,
    compute_effective_policy_hash,
)
from atagia.models.schemas_memory import RetrievalProfileId
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMProvider,
    StructuredOutputError,
)

MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"
OPERATIONAL_PROFILE = {
    "profile_id": "normal",
    "signals": {},
    "risk_level": "normal",
    "authorized": True,
    "profile_hash": "profile123",
    "token": "token123",
}


class StubStructuredProvider(LLMProvider):
    name = "stub-structured"

    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=self.output_text,
        )


def _resolved_policy(mode_id: str = "general_qa"):
    manifest = ManifestLoader(MANIFESTS_DIR).load_all()[mode_id]
    return PolicyResolver().resolve(manifest, None, None)


def _entry_payload(
    *,
    mode_id: str = "general_qa",
    prompt_hash: str | None = None,
    contract: dict[str, dict[str, object]] | None = None,
    cached_at: str = "2026-04-02T12:00:00+00:00",
    last_user_message_text: str = "Let's fix the failing login test",
    last_retrieval_message_seq: int = 10,
    workspace_id: str | None = None,
) -> dict[str, object]:
    resolved_policy = _resolved_policy(mode_id)
    return {
        "version": 2,
        "cache_key": "ctx:v2:test",
        "user_id": "usr_1",
        "conversation_id": "cnv_1",
        "assistant_mode_id": mode_id,
        "policy_prompt_hash": prompt_hash or resolved_policy.prompt_hash,
        "effective_policy_hash": compute_effective_policy_hash(resolved_policy),
        "operational_profile": OPERATIONAL_PROFILE,
        "workspace_id": workspace_id,
        "composed_context": {
            "contract_block": "Direct and concise.",
            "workspace_block": "",
            "memory_block": "Relevant memory block.",
            "state_block": "",
            "selected_memory_ids": ["mem_1"],
            "total_tokens_estimate": 60,
            "budget_tokens": 500,
            "items_included": 1,
            "items_dropped": 0,
        },
        "contract": contract or {},
        "memory_summaries": [
            {
                "memory_id": "mem_1",
                "text": "User prefers concise answers.",
                "object_type": "interaction_contract",
                "score": 0.91,
                "scope": "assistant_mode",
            }
        ],
        "detected_needs": ["ambiguity"],
        "source_retrieval_plan": {"fts_queries": ["login test"]},
        "selected_memory_ids": ["mem_1"],
        "cached_at": cached_at,
        "last_retrieval_message_seq": last_retrieval_message_seq,
        "last_user_message_text": last_user_message_text,
        "source": "sync",
    }


def _request_payload(
    *,
    message_text: str,
    mode_id: str = "general_qa",
    current_message_seq: int = 11,
    workspace_id: str | None = None,
    cache_enabled: bool = True,
    benchmark_mode: bool = False,
    replay_mode: bool = False,
    evaluation_mode: bool = False,
    mcp_mode: bool = False,
) -> dict[str, object]:
    return {
        "user_id": "usr_1",
        "conversation_id": "cnv_1",
        "workspace_id": workspace_id,
        "message_text": message_text,
        "current_message_seq": current_message_seq,
        "cache_enabled": cache_enabled,
        "operational_profile": OPERATIONAL_PROFILE,
        "effective_policy_hash": compute_effective_policy_hash(_resolved_policy(mode_id)),
        "benchmark_mode": benchmark_mode,
        "replay_mode": replay_mode,
        "evaluation_mode": evaluation_mode,
        "mcp_mode": mcp_mode,
    }


def _staleness_client(output_text: str) -> tuple[LLMClient[object], StubStructuredProvider]:
    provider = StubStructuredProvider(output_text)
    client = LLMClient(provider_name=provider.name, providers=[provider])
    return client, provider


@pytest.mark.asyncio
async def test_low_staleness_same_topic_continuation_stays_cacheable() -> None:
    client, provider = _staleness_client(
        """
        {
          "contradiction_detected": false,
          "high_stakes_topic": false,
          "sensitive_content": false,
          "mode_shift_target": null,
          "short_followup": true,
          "ambiguous_wording": false,
          "rationale": "Provider-specific explanation field."
        }
        """
    )
    scorer = ContextStalenessScorer(
        FrozenClock(datetime(2026, 4, 2, 12, 0, tzinfo=timezone.utc)),
        llm_client=client,
    )
    resolved_policy = _resolved_policy("coding_debug")

    score = await scorer.score(
        _entry_payload(mode_id="coding_debug"),
        _request_payload(
            message_text="Continue with the login test",
            mode_id="coding_debug",
        ),
        resolved_policy,
    )

    assert score.hard_sync is False
    assert score.should_refresh is False
    assert score.staleness < resolved_policy.context_cache_policy.sync_threshold
    assert score.token_overlap_ratio >= 0.6
    assert score.short_followup is True
    assert provider.requests[0].metadata["purpose"] == "context_cache_signal_detection"
    assert html.escape("Let's fix the failing login test") in provider.requests[0].messages[1].content
    assert html.escape("Continue with the login test") in provider.requests[0].messages[1].content
    assert RetrievalProfileId.CODING_DEBUG.value in provider.requests[0].messages[1].content


@pytest.mark.asyncio
async def test_multilingual_overlap_stays_cacheable_without_stopword_lists() -> None:
    client, provider = _staleness_client(
        """
        {
          "contradiction_detected": false,
          "high_stakes_topic": false,
          "sensitive_content": false,
          "mode_shift_target": null,
          "short_followup": false,
          "ambiguous_wording": false
        }
        """
    )
    scorer = ContextStalenessScorer(
        FrozenClock(datetime(2026, 4, 2, 12, 0, tzinfo=timezone.utc)),
        llm_client=client,
    )
    resolved_policy = _resolved_policy("general_qa")

    score = await scorer.score(
        _entry_payload(last_user_message_text="Vamos a arreglar la prueba de acceso"),
        _request_payload(message_text="Seguimos con la prueba de acceso"),
        resolved_policy,
    )

    assert score.hard_sync is False
    assert score.should_refresh is False
    assert score.token_overlap_ratio == pytest.approx(2 / 3)
    assert "high_token_overlap" in score.matched_signals
    assert provider.requests[0].metadata["purpose"] == "context_cache_signal_detection"


@pytest.mark.asyncio
async def test_contradiction_language_forces_sync() -> None:
    client, provider = _staleness_client(
        '{"contradiction_detected": true, "high_stakes_topic": false, "sensitive_content": false, "mode_shift_target": null, "short_followup": false, "ambiguous_wording": false}'
    )
    scorer = ContextStalenessScorer(
        FrozenClock(datetime(2026, 4, 2, 12, 1, tzinfo=timezone.utc)),
        llm_client=client,
    )
    resolved_policy = _resolved_policy("general_qa")

    score = await scorer.score(
        _entry_payload(),
        _request_payload(message_text="Actually, that changed and we should do the opposite."),
        resolved_policy,
    )

    assert score.hard_sync is True
    assert score.should_refresh is True
    assert "contradiction_language" in score.matched_signals
    assert provider.requests[0].metadata["purpose"] == "context_cache_signal_detection"


@pytest.mark.asyncio
async def test_explicit_mode_shift_forces_sync() -> None:
    client, provider = _staleness_client(
        '{"contradiction_detected": false, "high_stakes_topic": false, "sensitive_content": false, "mode_shift_target": "research_deep_dive", "short_followup": false, "ambiguous_wording": false}'
    )
    scorer = ContextStalenessScorer(
        FrozenClock(datetime(2026, 4, 2, 12, 1, tzinfo=timezone.utc)),
        llm_client=client,
    )
    resolved_policy = _resolved_policy("coding_debug")

    score = await scorer.score(
        _entry_payload(mode_id="coding_debug"),
        _request_payload(
            message_text="Switch to research mode and compare the papers.",
            mode_id="coding_debug",
        ),
        resolved_policy,
    )

    assert score.hard_sync is True
    assert score.should_refresh is True
    assert "mode_shift_language" in score.matched_signals
    assert any(signal.startswith("mode_shift_target:") for signal in score.matched_signals)
    assert provider.requests[0].messages[1].content.count("research_deep_dive") >= 1


@pytest.mark.asyncio
async def test_high_stakes_language_forces_sync() -> None:
    client, _ = _staleness_client(
        '{"contradiction_detected": false, "high_stakes_topic": true, "sensitive_content": false, "mode_shift_target": null, "short_followup": false, "ambiguous_wording": false}'
    )
    scorer = ContextStalenessScorer(
        FrozenClock(datetime(2026, 4, 2, 12, 1, tzinfo=timezone.utc)),
        llm_client=client,
    )
    resolved_policy = _resolved_policy("general_qa")

    # Token overlap is in the middle band (> deterministic-refresh max,
    # < deterministic-reuse min) so the scorer consults the LLM, whose
    # high_stakes_topic signal then forces a hard sync.
    score = await scorer.score(
        _entry_payload(
            last_user_message_text="We were discussing the medication dosage schedule earlier."
        ),
        _request_payload(
            message_text="I need medical advice about the medication dosage today."
        ),
        resolved_policy,
    )

    assert score.decision_band == "llm"
    assert score.hard_sync is True
    assert "high_stakes_language" in score.matched_signals


@pytest.mark.asyncio
async def test_sensitive_language_forces_sync() -> None:
    client, _ = _staleness_client(
        '{"contradiction_detected": false, "high_stakes_topic": false, "sensitive_content": true, "mode_shift_target": null, "short_followup": false, "ambiguous_wording": false}'
    )
    scorer = ContextStalenessScorer(
        FrozenClock(datetime(2026, 4, 2, 12, 1, tzinfo=timezone.utc)),
        llm_client=client,
    )
    resolved_policy = _resolved_policy("general_qa")

    # Token overlap is in the middle band so the scorer consults the LLM,
    # whose sensitive_content signal then forces a hard sync.
    score = await scorer.score(
        _entry_payload(
            last_user_message_text="I had a question about my bank account password earlier."
        ),
        _request_payload(message_text="Here is my password and bank account issue."),
        resolved_policy,
    )

    assert score.decision_band == "llm"
    assert score.hard_sync is True
    assert "sensitive_language" in score.matched_signals


@pytest.mark.asyncio
async def test_fast_pace_label_relaxes_hard_message_ceiling() -> None:
    client, provider = _staleness_client(
        '{"contradiction_detected": false, "high_stakes_topic": false, "sensitive_content": false, "mode_shift_target": null, "short_followup": true, "ambiguous_wording": false}'
    )
    scorer = ContextStalenessScorer(
        FrozenClock(datetime(2026, 4, 2, 12, 1, tzinfo=timezone.utc)),
        llm_client=client,
    )
    resolved_policy = _resolved_policy("general_qa")

    score = await scorer.score(
        _entry_payload(contract={"pace": {"label": "fast"}}),
        _request_payload(
            message_text="continue",
            current_message_seq=19,
        ),
        resolved_policy,
    )

    assert score.pace_label == "fast"
    assert score.pace_multiplier == 1.25
    assert score.effective_max_messages_without_refresh == 10
    assert score.hard_sync is False
    assert provider.requests[0].messages[1].content.startswith("You are deciding")


@pytest.mark.asyncio
async def test_unknown_pace_label_falls_back_to_default_multiplier() -> None:
    client, provider = _staleness_client(
        '{"contradiction_detected": false, "high_stakes_topic": false, "sensitive_content": false, "mode_shift_target": null, "short_followup": true, "ambiguous_wording": false}'
    )
    scorer = ContextStalenessScorer(
        FrozenClock(datetime(2026, 4, 2, 12, 1, tzinfo=timezone.utc)),
        llm_client=client,
    )
    resolved_policy = _resolved_policy("general_qa")

    score = await scorer.score(
        _entry_payload(contract={"pace": {"label": "turbo"}}),
        _request_payload(
            message_text="continue",
            current_message_seq=19,
        ),
        resolved_policy,
    )

    assert score.pace_label == "turbo"
    assert score.pace_multiplier == 1.0
    assert score.hard_sync is True
    assert "message_ceiling_exceeded" in score.matched_signals
    assert provider.requests == []


@pytest.mark.asyncio
async def test_missing_pace_label_uses_default_multiplier() -> None:
    client, _ = _staleness_client(
        '{"contradiction_detected": false, "high_stakes_topic": false, "sensitive_content": false, "mode_shift_target": null, "short_followup": true, "ambiguous_wording": false}'
    )
    scorer = ContextStalenessScorer(
        FrozenClock(datetime(2026, 4, 2, 12, 1, tzinfo=timezone.utc)),
        llm_client=client,
    )
    resolved_policy = _resolved_policy("general_qa")

    score = await scorer.score(
        _entry_payload(contract={}),
        _request_payload(message_text="continue"),
        resolved_policy,
    )

    assert score.pace_label == "default"
    assert score.pace_multiplier == 1.0


@pytest.mark.asyncio
async def test_ambiguous_short_wording_biases_to_sync_without_hard_override() -> None:
    client, provider = _staleness_client(
        '{"contradiction_detected": false, "high_stakes_topic": false, "sensitive_content": false, "mode_shift_target": null, "short_followup": false, "ambiguous_wording": true}'
    )
    scorer = ContextStalenessScorer(
        FrozenClock(datetime(2026, 4, 2, 12, 0, tzinfo=timezone.utc)),
        llm_client=client,
    )
    resolved_policy = _resolved_policy("general_qa")

    score = await scorer.score(
        _entry_payload(last_user_message_text="Let's fix the failing login test"),
        _request_payload(message_text="What about this?"),
        resolved_policy,
    )

    assert score.hard_sync is False
    assert score.should_refresh is True
    assert "ambiguous_wording" in score.matched_signals
    assert score.safety_penalty >= resolved_policy.context_cache_policy.sync_threshold
    assert provider.requests[0].metadata["purpose"] == "context_cache_signal_detection"


@pytest.mark.asyncio
async def test_cache_validation_failure_forces_sync_without_llm_call() -> None:
    client, provider = _staleness_client(
        '{"contradiction_detected": false, "high_stakes_topic": false, "sensitive_content": false, "mode_shift_target": null, "short_followup": false, "ambiguous_wording": false}'
    )
    scorer = ContextStalenessScorer(
        FrozenClock(datetime(2026, 4, 2, 12, 0, tzinfo=timezone.utc)),
        llm_client=client,
    )
    resolved_policy = _resolved_policy("general_qa")

    score = await scorer.score(
        {"cache_key": "ctx:v1:test"},
        _request_payload(message_text="Continue"),
        resolved_policy,
    )

    assert score.hard_sync is True
    assert "cache_entry_validation_failed" in score.matched_signals
    assert provider.requests == []


@pytest.mark.asyncio
async def test_invalid_mode_shift_target_fails_structured_validation() -> None:
    client, _ = _staleness_client(
        '{"contradiction_detected": false, "high_stakes_topic": false, "sensitive_content": false, "mode_shift_target": "not-a-real-mode", "short_followup": false, "ambiguous_wording": false}'
    )
    scorer = ContextStalenessScorer(
        FrozenClock(datetime(2026, 4, 2, 12, 0, tzinfo=timezone.utc)),
        llm_client=client,
    )
    resolved_policy = _resolved_policy("general_qa")

    with pytest.raises(StructuredOutputError):
        await scorer.score(
            _entry_payload(),
            _request_payload(message_text="Switch modes."),
            resolved_policy,
        )


# --- F2.2 deterministic-first staleness bands -------------------------------

_NO_SIGNALS_JSON = (
    '{"contradiction_detected": false, "high_stakes_topic": false, '
    '"sensitive_content": false, "mode_shift_target": null, '
    '"short_followup": false, "ambiguous_wording": false}'
)
# Cached previous message used by the deterministic-band fixtures. The reuse
# message below is a strict token subset of this (overlap 1.0 >= the HIGH
# threshold); the "below HIGH" flip message overlaps it at ~0.8 (< HIGH).
_BAND_PREVIOUS_MESSAGE = (
    "Can you walk me through the database migration steps again please"
)
_REUSE_MESSAGE = "Can you walk me through the database migration steps again"
_REFRESH_LONG_LOW_OVERLAP_MESSAGE = (
    "Switch topic completely and explain quantum entanglement basics now"
)


@pytest.mark.asyncio
async def test_deterministic_reuse_band_skips_llm_when_all_conditions_hold() -> None:
    client, provider = _staleness_client(_NO_SIGNALS_JSON)
    scorer = ContextStalenessScorer(
        FrozenClock(datetime(2026, 4, 2, 12, 0, tzinfo=timezone.utc)),
        llm_client=client,
    )
    resolved_policy = _resolved_policy("general_qa")

    score = await scorer.score(
        _entry_payload(last_user_message_text=_BAND_PREVIOUS_MESSAGE),
        _request_payload(message_text=_REUSE_MESSAGE, current_message_seq=11),
        resolved_policy,
    )

    assert provider.requests == []
    assert score.decision_band == "deterministic_reuse"
    assert score.hard_sync is False
    assert score.should_refresh is False
    assert score.token_overlap_ratio >= 0.85
    assert "high_token_overlap" in score.matched_signals


@pytest.mark.asyncio
async def test_deterministic_reuse_band_yields_to_llm_when_overlap_below_high() -> None:
    client, provider = _staleness_client(_NO_SIGNALS_JSON)
    scorer = ContextStalenessScorer(
        FrozenClock(datetime(2026, 4, 2, 12, 0, tzinfo=timezone.utc)),
        llm_client=client,
    )
    resolved_policy = _resolved_policy("general_qa")

    score = await scorer.score(
        _entry_payload(last_user_message_text=_BAND_PREVIOUS_MESSAGE),
        _request_payload(
            message_text="Walk me through the database migration steps once more please",
            current_message_seq=11,
        ),
        resolved_policy,
    )

    assert len(provider.requests) == 1
    assert 0.1 < score.token_overlap_ratio < 0.85
    assert score.decision_band == "llm"


@pytest.mark.asyncio
async def test_deterministic_reuse_band_yields_to_llm_when_two_messages_since_refresh() -> None:
    client, provider = _staleness_client(_NO_SIGNALS_JSON)
    scorer = ContextStalenessScorer(
        FrozenClock(datetime(2026, 4, 2, 12, 0, tzinfo=timezone.utc)),
        llm_client=client,
    )
    resolved_policy = _resolved_policy("general_qa")

    score = await scorer.score(
        _entry_payload(last_user_message_text=_BAND_PREVIOUS_MESSAGE),
        # entry.last_retrieval_message_seq == 10 => seq 12 is 2 messages since.
        _request_payload(message_text=_REUSE_MESSAGE, current_message_seq=12),
        resolved_policy,
    )

    assert len(provider.requests) == 1
    assert score.messages_since_refresh == 2
    assert score.decision_band == "llm"


@pytest.mark.asyncio
async def test_deterministic_reuse_band_yields_to_llm_when_age_beyond_fraction() -> None:
    # general_qa: max_minutes_without_refresh == 20, age fraction 0.25 => 5.0 min
    # boundary. 6 minutes is past the reuse fraction but under the hard ceiling.
    client, provider = _staleness_client(_NO_SIGNALS_JSON)
    scorer = ContextStalenessScorer(
        FrozenClock(datetime(2026, 4, 2, 12, 6, tzinfo=timezone.utc)),
        llm_client=client,
    )
    resolved_policy = _resolved_policy("general_qa")

    score = await scorer.score(
        _entry_payload(last_user_message_text=_BAND_PREVIOUS_MESSAGE),
        _request_payload(message_text=_REUSE_MESSAGE, current_message_seq=11),
        resolved_policy,
    )

    assert len(provider.requests) == 1
    assert score.minutes_since_refresh == pytest.approx(6.0)
    assert score.decision_band == "llm"


@pytest.mark.asyncio
async def test_deterministic_reuse_band_yields_to_llm_when_message_is_short() -> None:
    # A high-overlap but SHORT message is excluded from the reuse band: the
    # asymmetric overlap ratio is uninformative for terse follow-ups, so the
    # LLM's short_followup signal handles it instead.
    client, provider = _staleness_client(_NO_SIGNALS_JSON)
    scorer = ContextStalenessScorer(
        FrozenClock(datetime(2026, 4, 2, 12, 0, tzinfo=timezone.utc)),
        llm_client=client,
    )
    resolved_policy = _resolved_policy("general_qa")

    score = await scorer.score(
        _entry_payload(last_user_message_text=_BAND_PREVIOUS_MESSAGE),
        # "database migration" is a strict subset of the cached message
        # (overlap 1.0) but only 2 tokens long, so it is treated as short.
        _request_payload(message_text="database migration", current_message_seq=11),
        resolved_policy,
    )

    assert len(provider.requests) == 1
    assert score.token_overlap_ratio >= 0.85
    assert score.decision_band == "llm"


@pytest.mark.asyncio
async def test_deterministic_refresh_band_skips_llm_for_long_low_overlap_message() -> None:
    client, provider = _staleness_client(_NO_SIGNALS_JSON)
    scorer = ContextStalenessScorer(
        FrozenClock(datetime(2026, 4, 2, 12, 0, tzinfo=timezone.utc)),
        llm_client=client,
    )
    resolved_policy = _resolved_policy("general_qa")

    score = await scorer.score(
        _entry_payload(last_user_message_text=_BAND_PREVIOUS_MESSAGE),
        _request_payload(
            message_text=_REFRESH_LONG_LOW_OVERLAP_MESSAGE,
            current_message_seq=11,
        ),
        resolved_policy,
    )

    assert provider.requests == []
    assert score.decision_band == "deterministic_refresh"
    assert score.hard_sync is False
    assert score.should_refresh is True
    assert score.token_overlap_ratio <= 0.1
    assert "low_token_overlap" in score.matched_signals


@pytest.mark.asyncio
async def test_low_overlap_short_message_yields_to_llm_asymmetry_caveat() -> None:
    # A SHORT low-overlap message must NOT trigger the deterministic refresh
    # band: a terse continuation scores low overlap by construction, so it is
    # routed to the LLM (whose short_followup signal can still keep the cache).
    client, provider = _staleness_client(
        '{"contradiction_detected": false, "high_stakes_topic": false, '
        '"sensitive_content": false, "mode_shift_target": null, '
        '"short_followup": true, "ambiguous_wording": false}'
    )
    scorer = ContextStalenessScorer(
        FrozenClock(datetime(2026, 4, 2, 12, 0, tzinfo=timezone.utc)),
        llm_client=client,
    )
    resolved_policy = _resolved_policy("general_qa")

    score = await scorer.score(
        _entry_payload(last_user_message_text=_BAND_PREVIOUS_MESSAGE),
        _request_payload(message_text="and now?", current_message_seq=11),
        resolved_policy,
    )

    assert len(provider.requests) == 1
    assert score.token_overlap_ratio <= 0.1
    assert score.decision_band == "llm"
    assert score.short_followup is True


@pytest.mark.asyncio
async def test_middle_band_message_calls_llm_and_records_band() -> None:
    # Overlap strictly between the two deterministic thresholds => LLM path,
    # unchanged from before, with the band recorded as "llm".
    client, provider = _staleness_client(_NO_SIGNALS_JSON)
    scorer = ContextStalenessScorer(
        FrozenClock(datetime(2026, 4, 2, 12, 0, tzinfo=timezone.utc)),
        llm_client=client,
    )
    resolved_policy = _resolved_policy("general_qa")

    score = await scorer.score(
        _entry_payload(last_user_message_text="Vamos a arreglar la prueba de acceso"),
        _request_payload(message_text="Seguimos con la prueba de acceso"),
        resolved_policy,
    )

    assert len(provider.requests) == 1
    assert 0.1 < score.token_overlap_ratio < 0.85
    assert score.decision_band == "llm"
