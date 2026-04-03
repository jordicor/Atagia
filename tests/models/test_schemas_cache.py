"""Tests for adaptive context-cache schemas."""

from __future__ import annotations

import pytest

from atagia.models.schemas_cache import ContextCacheEntry


def _cache_entry_payload() -> dict[str, object]:
    return {
        "cache_key": "ctx:v1:test",
        "user_id": "usr_1",
        "conversation_id": "cnv_1",
        "assistant_mode_id": "coding_debug",
        "policy_prompt_hash": "abc123",
        "workspace_id": "wrk_1",
        "composed_context": {
            "contract_block": "Direct, concise.",
            "workspace_block": "Workspace notes.",
            "memory_block": "Relevant memories.",
            "state_block": "",
            "selected_memory_ids": ["mem_1", "mem_2"],
            "total_tokens_estimate": 120,
            "budget_tokens": 600,
            "items_included": 2,
            "items_dropped": 0,
        },
        "contract": {"pace": {"label": "fast"}},
        "memory_summaries": [
            {
                "memory_id": "mem_1",
                "text": "User prefers direct answers.",
                "object_type": "interaction_contract",
                "score": 0.92,
                "scope": "assistant_mode",
            }
        ],
        "detected_needs": ["ambiguity"],
        "source_retrieval_plan": {"fts_queries": ["direct answers"]},
        "selected_memory_ids": ["mem_1", "mem_2"],
        "cached_at": "2026-04-02T12:00:00+00:00",
        "last_retrieval_message_seq": 8,
        "last_user_message_text": "Can you keep it concise?",
        "source": "sync",
    }


def test_context_cache_entry_validates_successfully() -> None:
    entry = ContextCacheEntry.model_validate(_cache_entry_payload())

    assert entry.version == 1
    assert entry.composed_context.selected_memory_ids == ["mem_1", "mem_2"]
    assert entry.memory_summaries[0].memory_id == "mem_1"


def test_context_cache_entry_rejects_duplicate_detected_needs() -> None:
    payload = _cache_entry_payload()
    payload["detected_needs"] = ["ambiguity", "ambiguity"]

    with pytest.raises(ValueError):
        ContextCacheEntry.model_validate(payload)


def test_context_cache_entry_rejects_duplicate_selected_memory_ids() -> None:
    payload = _cache_entry_payload()
    payload["selected_memory_ids"] = ["mem_1", "mem_1"]

    with pytest.raises(ValueError):
        ContextCacheEntry.model_validate(payload)


def test_context_cache_entry_rejects_invalid_source() -> None:
    payload = _cache_entry_payload()
    payload["source"] = "cache_hit"

    with pytest.raises(ValueError):
        ContextCacheEntry.model_validate(payload)
