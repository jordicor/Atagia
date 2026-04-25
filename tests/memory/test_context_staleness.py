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
from atagia.models.schemas_memory import AssistantModeId
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
          "ambiguous_wording": false
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
    assert AssistantModeId.CODING_DEBUG.value in provider.requests[0].messages[1].content


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

    score = await scorer.score(
        _entry_payload(),
        _request_payload(message_text="I need medical advice about medication dosage today."),
        resolved_policy,
    )

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

    score = await scorer.score(
        _entry_payload(),
        _request_payload(message_text="Here is my password and bank account issue."),
        resolved_policy,
    )

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
