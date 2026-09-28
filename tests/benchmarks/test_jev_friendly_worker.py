"""Offline production-path check for isolated extraction and retrieval."""

from __future__ import annotations

import json
import re

import pytest

from atagia.core.source_references import SourceReferenceCatalog
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMProvider,
)
from benchmarks.jev_friendly_cards.worker import _context_messages, run_full_flow


class NoMemoryProvider(LLMProvider):
    name = "openrouter"

    def __init__(self) -> None:
        self.requests: list[LLMCompletionRequest] = []
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        if self.closed:
            raise AssertionError("A completed slot reused its closed provider")
        self.requests.append(request)
        outputs = {
            "memory_extraction_candidate_card": "none",
            "need_detection_needs_card": "none",
            "need_detection_query_language_card": "en",
            "need_detection_answer_language_card": "en",
            "need_detection_memory_card": "personal",
            "need_detection_exact_card": "no",
            "need_detection_shape_card": "broad",
            "need_detection_facets_card": "none",
            "need_detection_callback_card": "no",
            "need_detection_search_words_card": "thanks",
            "topic_working_set_route_card": "none",
        }
        purpose = request.metadata["purpose"]
        if purpose in {"applicability_relevance_card", "applicability_date_card"}:
            keys = re.findall(r'<candidate [^>]*score_key="([^"]+)"', request.messages[-1].content)
            output = "\n".join(
                f"{key} {'drop' if purpose == 'applicability_relevance_card' else 'none'}"
                for key in keys
            )
        elif purpose in outputs:
            output = outputs[purpose]
        else:
            raise AssertionError(f"Unexpected model call: {purpose}")
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=output,
            usage={"input_tokens": 20, "output_tokens": 2},
        )


class OneFactProvider(NoMemoryProvider):
    def __init__(self, source: str) -> None:
        super().__init__()
        self.last_ref = SourceReferenceCatalog(source).anchors[-1].reference_id

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        if self.closed:
            raise AssertionError("A completed slot reused its closed provider")
        purpose = request.metadata["purpose"]
        outputs = {
            "memory_extraction_candidate_card": "cand_001 | The speaker's default editor is Zed.",
            "memory_extraction_kind_card": "evidence",
            "memory_extraction_scope_card": "user",
            "memory_extraction_confidence_card": "0.83",
            "memory_extraction_evidence_support_card": "direct",
            "memory_extraction_preserve_verbatim_card": "yes",
            "memory_extraction_candidate_language_card": "en",
            "memory_extraction_source_reference_card": f"r1 {self.last_ref}",
            "memory_extraction_index_card": "cand_001 | default editor Zed",
            "memory_extraction_temporal_type_card": "none",
            "memory_extraction_coverage_members_card": "none",
            "intent_classifier_explicit": '{"is_explicit": true, "reasoning": "The user stated it."}',
        }
        if purpose == "applicability_relevance_card":
            self.requests.append(request)
            keys = re.findall(r'<candidate [^>]*score_key="([^"]+)"', request.messages[-1].content)
            output = "\n".join(f"{key} exact" for key in keys)
        elif purpose in outputs:
            self.requests.append(request)
            output = outputs[purpose]
        else:
            return await super().complete(request)
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=output,
            usage={"input_tokens": 20, "output_tokens": 2},
        )


def test_unavailable_real_context_is_omitted_without_inventing_event_time():
    messages, omitted, provenance = _context_messages({
        "origin": "aurvek_local_snapshot",
        "context_messages": [
            {"role": "user", "text_available": False, "stored_raw_message": "private"},
            {"role": "bot", "text_available": True, "text": "Available context."},
        ],
    })
    assert omitted == [{"original_index": 1, "role": "user"}]
    assert [(message.id, message.seq, message.role) for message in messages] == [
        ("recent_1", 1, "assistant")
    ]
    assert provenance == [
        {"original_index": 1, "original_role": "user", "atagia_role": "user", "included": False},
        {"original_index": 2, "original_role": "bot", "atagia_role": "assistant", "included": True},
    ]
    assert messages[0].content == "Available context."
    assert messages[0].occurred_at is None


@pytest.mark.parametrize("origin", ["synthetic", "aurvek_local_snapshot"])
def test_unknown_context_role_fails_even_when_text_is_unavailable(origin):
    with pytest.raises(ValueError, match="Unsupported .* message role"):
        _context_messages({
            "origin": origin,
            "context_messages": [
                {"role": "unknown", "text_available": False, "text": "Not used", "content": "Not used"},
            ],
        })


def test_aurvek_bot_role_is_not_accepted_in_synthetic_context():
    with pytest.raises(ValueError, match="Unsupported synthetic message role"):
        _context_messages({
            "origin": "synthetic",
            "context_messages": [{"role": "bot", "content": "Not an Atagia role"}],
        })


@pytest.mark.asyncio
async def test_no_memory_full_flow_retrieves_its_own_empty_sqlite_state(tmp_path):
    provider = NoMemoryProvider()
    client = LLMClient(providers=[provider], structured_output_retry_attempts=0)
    case = {
        "case_id": "offline",
        "origin": "synthetic",
        "primary_family": "classification",
        "role": "user",
        "source_text": "Thanks, that answers my question.",
        "context_messages": [],
    }
    fixture = {
        "source_message_id": "offline",
        "user_id": "offline_user",
        "conversation_id": "offline_conversation",
        "assistant_mode_id": "general_qa",
        "mode": "general_qa",
        "occurred_at": "2026-09-26T12:00:00+00:00",
        "privacy_enforcement": "off",
    }
    result = await run_full_flow(
        client,
        case=case,
        fixture=fixture,
        followup_query="Do I have a remembered preference?",
        arm="B_shared_llm",
        database_path=tmp_path / "flow.sqlite",
    )
    assert result["extraction"]["nothing_durable"] is True
    assert result["database_rows"]["memory_objects"] == []
    assert result["retrieval"]["composed_context"] is not None
    assert result["retrieval_recent_message_ids"] == ["followup_query"]
    assert result["role_provenance"] == {
        "source": {"original_role": "user", "atagia_role": "user"},
        "context": [],
    }
    assert provider.closed
    assert {request.metadata["purpose"] for request in provider.requests} >= {
        "memory_extraction_candidate_card",
        "need_detection_query_language_card",
        "need_detection_answer_language_card",
    }


@pytest.mark.asyncio
async def test_fact_full_flow_persists_source_span_and_retrieves_own_memory(tmp_path):
    source = "My default editor is Zed."
    provider = OneFactProvider(source)
    client = LLMClient(providers=[provider], structured_output_retry_attempts=0)
    result = await run_full_flow(
        client,
        case={
            "case_id": "fact",
            "origin": "synthetic",
            "primary_family": "classification",
            "role": "user",
            "source_text": source,
            "context_messages": [],
        },
        fixture={
            "source_message_id": "fact",
            "user_id": "fact_user",
            "conversation_id": "fact_conversation",
            "assistant_mode_id": "general_qa",
            "mode": "general_qa",
            "occurred_at": "2026-09-26T12:00:00+00:00",
            "privacy_enforcement": "off",
        },
        followup_query="Which editor do I use by default?",
        arm="B_shared_llm",
        database_path=tmp_path / "fact.sqlite",
    )
    assert result["database_rows"]["memory_objects"]
    assert any(
        span["quote_text"] == source
        for span in result["database_rows"]["memory_evidence_spans"]
    )
    assert result["retrieval"]["raw_candidates"]
    memory_ids = {row["id"] for row in result["database_rows"]["memory_objects"]}
    assert memory_ids & {row["id"] for row in result["retrieval"]["raw_candidates"]}
    assert memory_ids & set(result["retrieval"]["composed_context"]["selected_memory_ids"])
    assert result["retrieval_recent_message_ids"] == ["followup_query"]
    assert result["retrieval"]["trace"]["candidate_search"] is not None
    assert json.loads(json.dumps(result))["retrieval_recent_message_ids"] == [
        "followup_query"
    ]
    assert provider.closed


@pytest.mark.asyncio
async def test_sequential_slots_use_fresh_owned_clients(tmp_path):
    case = {
        "case_id": "offline", "origin": "synthetic", "primary_family": "classification",
        "role": "user", "source_text": "Thanks, that answers my question.",
        "context_messages": [],
    }
    fixture = {
        "source_message_id": "offline", "user_id": "offline_user",
        "conversation_id": "offline_conversation", "assistant_mode_id": "general_qa",
        "mode": "general_qa", "occurred_at": "2026-09-26T12:00:00+00:00",
        "privacy_enforcement": "off",
    }
    providers = [NoMemoryProvider(), NoMemoryProvider()]
    for index, provider in enumerate(providers):
        await run_full_flow(
            LLMClient(providers=[provider], structured_output_retry_attempts=0),
            case=case, fixture=fixture,
            followup_query="Do I have a remembered preference?",
            arm="B_shared_llm", database_path=tmp_path / f"slot_{index}.sqlite",
        )
        assert provider.closed
    assert all(provider.requests for provider in providers)


@pytest.mark.asyncio
async def test_topic_no_change_full_flow_reaches_retrieval(tmp_path):
    provider = NoMemoryProvider()
    result = await run_full_flow(
        LLMClient(providers=[provider], structured_output_retry_attempts=0),
        case={
            "case_id": "topic", "origin": "synthetic", "primary_family": "topics",
            "role": "user", "source_text": "Okay, thanks.", "context_messages": [],
        },
        fixture={
            "source_message_id": "topic", "user_id": "topic_user",
            "conversation_id": "topic_conversation", "assistant_mode_id": "general_qa",
            "mode": "general_qa", "occurred_at": "2026-09-26T12:00:00+00:00",
            "privacy_enforcement": "off",
        },
        followup_query="What is the current topic?",
        arm="B_shared_llm", database_path=tmp_path / "topic.sqlite",
    )
    assert result["topic_updates"] == []
    assert result["retrieval"]["trace"] is not None
    json.dumps(result)
    assert sum(
        request.metadata["purpose"] == "topic_working_set_route_card"
        for request in provider.requests
    ) == 2
    assert provider.closed
