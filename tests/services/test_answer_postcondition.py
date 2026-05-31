"""Tests for guarded answer postconditions."""

from __future__ import annotations

import json

import pytest

from atagia.models.schemas_memory import ComposedContext
from atagia.services.answer_postcondition import (
    _abstain_result,
    complete_answer_with_postcondition_guard,
)
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMMessage,
    LLMProvider,
    OutputLimitExceededError,
)
from atagia.services.llm_reliability import LLMTechnicalRecoveryConfig
from atagia.services.prompt_authority import normalize_request_authority_context


class AnswerGuardProvider(LLMProvider):
    name = "answer-guard-tests"

    def __init__(
        self, outputs: list[str], verdicts: list[dict[str, object] | str]
    ) -> None:
        self.outputs = list(outputs)
        self.verdicts = list(verdicts)
        self.requests: list[LLMCompletionRequest] = []
        self.raise_output_limit_once = False
        self.raise_output_limit_on_retry_once = False

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        purpose = str(request.metadata.get("purpose"))
        if purpose in {
            "answer_postcondition_verification",
            "answer_abstention_legitimacy_verification",
            "answer_evidence_use_verification",
        }:
            verdict = self.verdicts.pop(0)
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=verdict
                if isinstance(verdict, str)
                else json.dumps(verdict),
            )
        if (
            self.raise_output_limit_on_retry_once
            and request.metadata.get("atagia_answer_postcondition_retry")
        ):
            self.raise_output_limit_on_retry_once = False
            raise OutputLimitExceededError("max output tokens")
        if self.raise_output_limit_once:
            self.raise_output_limit_once = False
            raise OutputLimitExceededError("max output tokens")
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=self.outputs.pop(0),
            usage={
                "prompt_tokens": 42,
                "completion_tokens": 9,
                "total_tokens": 51,
            },
            raw_response={"choices": [{"finish_reason": "stop"}]},
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError(f"Embeddings are not used here: {request.model}")


def _context() -> ComposedContext:
    return ComposedContext(
        memory_block="[Retrieved Memories]\n1. naproxen 500 mg twice a day",
        selected_memory_ids=["mem_1"],
        total_tokens_estimate=12,
        budget_tokens=100,
        items_included=1,
        items_dropped=0,
    )


def _secret_context() -> ComposedContext:
    return ComposedContext(
        memory_block="[Retrieved Memories]\n1. production jump host password: K8sN0d3Jump!2024",
        selected_memory_ids=["mem_secret"],
        total_tokens_estimate=12,
        budget_tokens=100,
        items_included=1,
        items_dropped=0,
    )


def _source_time_private_context() -> ComposedContext:
    return ComposedContext(
        memory_block=(
            "[Retrieved Memories]\n"
            "1. Ben is seeing Dr. Reeves for anxiety. "
            "At the time, he said: do not use this in any other context; "
            "it is private to this conversation."
        ),
        selected_memory_ids=["mem_1"],
        total_tokens_estimate=24,
        budget_tokens=100,
        items_included=1,
        items_dropped=0,
    )


def _answer_evidence_context() -> ComposedContext:
    return ComposedContext(
        memory_block=(
            "[Retrieved Memories]\n"
            "1. Jon talked about the studio opening and related celebration."
        ),
        selected_memory_ids=["mem_vibes"],
        answer_evidence_sufficiency={
            "state": "sufficient_direct_quote",
            "confidence": 0.96,
            "rendered": False,
            "top_memory_id": "mem_vibes",
        },
        answer_evidence_items=[
            {
                "memory_id": "mem_vibes",
                "claim": "Jon wanted to savor all the good vibes.",
                "supporting_quote": "Jon: I want to savor all the good vibes.",
                "quote_source": "source_message",
                "support_kind": "contextual_direct",
                "source_chain": [
                    "assistant seq 282: The studio looks amazing.",
                    "user seq 283: I want to savor all the good vibes.",
                ],
                "selected_for_answer_pack": False,
                "final_score": 0.96,
            }
        ],
        total_tokens_estimate=28,
        budget_tokens=100,
        items_included=1,
        items_dropped=0,
    )


def _answer_support_context() -> ComposedContext:
    return ComposedContext(
        memory_block=(
            "[Retrieved Memories]\n"
            "1. Caroline said Paris was one of the cities.\n"
            "2. Caroline said Rome was one of the cities."
        ),
        selected_memory_ids=["mem_paris", "mem_rome"],
        answer_shape="list",
        coverage_mode="exhaustive_known_set",
        source_precision="required",
        coverage_state="partial",
        support_map={
            "mem_paris": ["memory:mem_paris", "message:msg_paris"],
            "mem_rome": ["memory:mem_rome", "message:msg_rome"],
        },
        allowed_values=[
            {
                "display_text": "Paris",
                "normalized_key": "value|paris",
                "evidence_ids": ["memory:mem_paris", "message:msg_paris"],
                "memory_ids": ["mem_paris"],
            },
            {
                "display_text": "Rome",
                "normalized_key": "value|rome",
                "evidence_ids": ["memory:mem_rome", "message:msg_rome"],
                "memory_ids": ["mem_rome"],
            },
        ],
        missing_slots=[
            {
                "normalized_key": "value|unknown",
                "reason": "source_backed_group_not_selected",
            }
        ],
        total_tokens_estimate=32,
        budget_tokens=100,
        items_included=2,
        items_dropped=1,
    )


def _supported_retrieval_diagnostics() -> dict[str, object]:
    return {
        "need_detection": {
            "query_type": "slot_fill",
            "exact_recall_needed": True,
        },
        "retrieval_sufficiency": {
            "state": "retrieval_sufficient",
            "would_abstain": False,
        },
        "facet_support": {
            "obligations": [
                {
                    "id": "f1",
                    "description": "What medication did Rosa mention?",
                    "status": "covered",
                    "selected_memory_ids": ["mem_1"],
                    "composed_memory_ids": ["mem_1"],
                    "support_verdict": "supported",
                }
            ]
        },
        "direct_vs_indirect_provenance": {
            "evidence": [
                {
                    "memory_id": "mem_1",
                    "recovery_channels": ["fts"],
                    "proof_source": "base_canonical",
                    "joined_to_base": True,
                    "selected": True,
                    "matched_subquery_indexes": [0],
                }
            ],
            "direct_recovery_count": 1,
            "indirect_recovery_count": 0,
            "summary_only_count": 0,
            "raw_cross_conversation_count": 0,
        },
        "cross_conversation_raw_policy": {
            "enabled": False,
            "violation_count": 0,
        },
    }


def _answer_evidence_retrieval_diagnostics() -> dict[str, object]:
    return {
        "need_detection": {
            "query_type": "temporal",
            "exact_recall_needed": False,
        },
        "retrieval_sufficiency": {
            "state": "retrieval_sufficient",
            "would_abstain": False,
        },
        "answer_evidence": {
            "sufficiency": {
                "state": "sufficient_direct_quote",
                "confidence": 0.96,
                "rendered": False,
                "top_memory_id": "mem_vibes",
            },
            "items": [
                {
                    "memory_id": "mem_vibes",
                    "supporting_quote": "Jon: I want to savor all the good vibes.",
                    "quote_source": "source_message",
                    "support_kind": "contextual_direct",
                    "selected_for_answer_pack": False,
                }
            ],
            "direct_memory_ids": ["mem_vibes"],
        },
        "cross_conversation_raw_policy": {
            "enabled": False,
            "violation_count": 0,
        },
    }


def _broad_list_answer_evidence_retrieval_diagnostics() -> dict[str, object]:
    diagnostics = _answer_evidence_retrieval_diagnostics()
    diagnostics["need_detection"] = {
        "query_type": "broad_list",
        "exact_recall_needed": True,
    }
    diagnostics["answer_evidence"] = {
        "sufficiency": {
            "state": "sufficient_direct_quote",
            "confidence": 0.90,
            "rendered": False,
            "top_memory_id": "sum_rome",
        },
        "items": [
            {
                "memory_id": "sum_rome",
                "supporting_quote": (
                    "Jon: Took a short trip last week to Rome to clear my mind."
                ),
                "quote_source": "evidence_packet_source",
                "support_kind": "inferred",
                "selected_for_answer_pack": False,
            },
            {
                "memory_id": "vew_paris",
                "supporting_quote": "Jon: I've been to Paris yesterday!",
                "quote_source": "source_message",
                "support_kind": "",
                "selected_for_answer_pack": False,
            },
        ],
    }
    return diagnostics


def _supported_retrieval_plan_diagnostics() -> dict[str, object]:
    diagnostics = dict(_supported_retrieval_diagnostics())
    diagnostics.pop("need_detection")
    diagnostics["retrieval_plan"] = {
        "query_type": "slot_fill",
        "exact_recall_mode": True,
        "raw_context_access_mode": "normal",
        "retrieval_levels": [0],
    }
    diagnostics["diagnostic_shape_fallback_used"] = True
    return diagnostics


def _request() -> LLMCompletionRequest:
    return LLMCompletionRequest(
        model="openai/gpt-5-mini",
        messages=[
            LLMMessage(role="system", content="Answer from memory."),
            LLMMessage(role="user", content="What medication did Rosa mention?"),
        ],
        metadata={"purpose": "chat_reply"},
        max_output_tokens=8192,
    )


def _pass_verdict() -> dict[str, object]:
    return {
        "readable": True,
        "is_abstention": False,
        "contains_concrete_claims": True,
        "unsupported_concrete_claims": False,
        "covers_requested_facets": True,
        "requires_abstention": False,
        "pass_postcondition": True,
        "failure_reasons": [],
        "explanation": "Supported by the retrieved memory.",
    }


@pytest.mark.asyncio
async def test_answer_postcondition_passes_supported_answer() -> None:
    provider = AnswerGuardProvider(
        outputs=["Rosa mentioned naproxen 500 mg twice a day."],
        verdicts=[_pass_verdict()],
    )
    result = await complete_answer_with_postcondition_guard(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        request=_request(),
        verifier_model="openai/gpt-5-mini",
        original_query="What medication did Rosa mention?",
        composed_context=_context(),
    )

    assert result.output_text == "Rosa mentioned naproxen 500 mg twice a day."
    assert result.report.status == "passed"
    assert result.report.retry_count == 0
    assert result.report.token_budget is not None
    assert result.report.token_budget.prompt_tokens == 42
    assert result.report.token_budget.completion_tokens == 9
    assert result.report.token_budget.context_tokens == 12
    assert result.report.token_budget.answer_max_tokens == 8192
    assert result.report.token_budget.finish_reason == "stop"
    verifier_request = next(
        request
        for request in provider.requests
        if request.metadata.get("purpose") == "answer_postcondition_verification"
    )
    assert "substantive response is that the requested fact cannot be provided" in (
        verifier_request.messages[1].content
    )


@pytest.mark.asyncio
async def test_answer_postcondition_verifier_receives_answer_stance() -> None:
    provider = AnswerGuardProvider(
        outputs=[
            (
                "No allergies are explicitly recorded for Rosa. The retrieved "
                "context says ibuprofen caused stomach upset, but that is not "
                "the same as a confirmed allergy."
            )
        ],
        verdicts=[_pass_verdict()],
    )
    result = await complete_answer_with_postcondition_guard(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        request=_request(),
        verifier_model="openai/gpt-5-mini",
        original_query="Does Rosa have any allergies?",
        composed_context=_context(),
        answer_stance="proactive",
    )

    assert result.report.status == "passed"
    verifier_request = next(
        request
        for request in provider.requests
        if request.metadata.get("purpose") == "answer_postcondition_verification"
    )
    assert verifier_request.metadata["answer_stance"] == "proactive"
    assert "answer_stance=proactive" in verifier_request.messages[1].content
    assert "related, not the same fact" in (
        verifier_request.messages[1].content
    )


@pytest.mark.asyncio
async def test_answer_postcondition_verifier_receives_answer_evidence() -> None:
    provider = AnswerGuardProvider(
        outputs=["Jon wanted to make awesome memories."],
        verdicts=[
            _pass_verdict(),
            {
                "uses_required_evidence": True,
                "should_repair": False,
                "reason": "The answer uses the required evidence.",
                "missing_supported_obligations": [],
                "evidence_ids_supporting_answer": ["mem_vibes"],
            },
        ],
    )
    result = await complete_answer_with_postcondition_guard(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        request=_request(),
        verifier_model="openai/gpt-5-mini",
        original_query="What did Jon want to savor?",
        composed_context=_answer_evidence_context(),
        privacy_enforcement="off",
    )

    assert result.report.status == "passed"
    verifier_request = next(
        request
        for request in provider.requests
        if request.metadata.get("purpose") == "answer_postcondition_verification"
    )
    verifier_prompt = verifier_request.messages[1].content
    assert '"answer_evidence"' in verifier_prompt
    assert "sufficient_direct_quote" in verifier_prompt
    assert "Jon: I want to savor all the good vibes." in verifier_prompt
    assert "conflicting nearby answer" in verifier_prompt


@pytest.mark.asyncio
async def test_answer_postcondition_repairs_answer_that_ignores_answer_evidence() -> (
    None
):
    provider = AnswerGuardProvider(
        outputs=[
            "Jon wanted to make awesome memories.",
            "Jon wanted to savor all the good vibes.",
        ],
        verdicts=[
            _pass_verdict(),
            {
                "uses_required_evidence": False,
                "should_repair": True,
                "reason": "The answer replaced the direct quote with a nearby phrase.",
                "missing_supported_obligations": [],
                "evidence_ids_supporting_answer": ["mem_vibes"],
            },
            _pass_verdict(),
        ],
    )

    result = await complete_answer_with_postcondition_guard(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        request=_request(),
        verifier_model="openai/gpt-5-mini",
        original_query="What did Jon want to savor?",
        composed_context=_answer_evidence_context(),
        retrieval_sufficiency={
            "state": "retrieval_sufficient",
            "would_abstain": False,
        },
        retrieval_diagnostics=_answer_evidence_retrieval_diagnostics(),
        privacy_enforcement="off",
    )

    assert result.output_text == "Jon wanted to savor all the good vibes."
    assert result.report.status == "retry_passed"
    assert result.report.evidence_use_repair_count == 1
    assert result.report.evidence_use_repair_success_count == 1
    assert result.report.final_answer_used_required_evidence is True
    evidence_use_request = next(
        request
        for request in provider.requests
        if request.metadata.get("purpose") == "answer_evidence_use_verification"
    )
    assert "different nearby phrase" in evidence_use_request.messages[0].content
    repair_request = next(
        request
        for request in provider.requests
        if request.metadata.get("atagia_answer_evidence_use_repair")
    )
    repair_payload = "; ".join(
        repair_request.metadata["atagia_answer_evidence_use_repair_obligations"]
    )
    assert "Jon: I want to savor all the good vibes." in repair_payload


@pytest.mark.asyncio
async def test_answer_postcondition_repairs_unsupported_value_using_answer_support() -> (
    None
):
    provider = AnswerGuardProvider(
        outputs=[
            "Caroline mentioned Paris, Rome, and downtown.",
            "Caroline mentioned Paris and Rome from the supported retrieved evidence.",
        ],
        verdicts=[
            {
                "readable": True,
                "is_abstention": False,
                "contains_concrete_claims": True,
                "unsupported_concrete_claims": True,
                "covers_requested_facets": False,
                "requires_abstention": False,
                "pass_postcondition": False,
                "failure_reasons": ["unsupported_concrete_claim"],
                "explanation": "The answer adds downtown, which is not allowed.",
            },
            _pass_verdict(),
        ],
    )

    result = await complete_answer_with_postcondition_guard(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        request=_request(),
        verifier_model="openai/gpt-5-mini",
        original_query="Which cities did Caroline mention?",
        composed_context=_answer_support_context(),
        privacy_enforcement="off",
    )

    assert result.output_text == (
        "Caroline mentioned Paris and Rome from the supported retrieved evidence."
    )
    assert result.report.status == "retry_passed"
    initial_answer_request = provider.requests[0]
    assert "<answer_support>" in initial_answer_request.messages[0].content
    assert '"allowed_values"' in initial_answer_request.messages[0].content
    verifier_request = next(
        request
        for request in provider.requests
        if request.metadata.get("purpose") == "answer_postcondition_verification"
    )
    assert '"answer_support"' in verifier_request.messages[1].content
    repair_request = next(
        request
        for request in provider.requests
        if request.metadata.get("atagia_answer_postcondition_retry")
    )
    assert "use only allowed_values" in repair_request.messages[0].content
    assert "<answer_support>" in repair_request.messages[0].content


@pytest.mark.asyncio
async def test_answer_postcondition_falls_back_to_supported_partial_answer() -> None:
    provider = AnswerGuardProvider(
        outputs=[
            "Caroline mentioned Paris, Rome, and downtown.",
            "Caroline mentioned Paris, Rome, and downtown again.",
            "Caroline mentioned Paris, Rome, and downtown once more.",
        ],
        verdicts=[
            {
                "readable": True,
                "is_abstention": False,
                "contains_concrete_claims": True,
                "unsupported_concrete_claims": True,
                "covers_requested_facets": False,
                "requires_abstention": False,
                "pass_postcondition": False,
                "failure_reasons": ["unsupported_concrete_claim"],
                "explanation": "The answer adds downtown.",
            },
            {
                "readable": True,
                "is_abstention": False,
                "contains_concrete_claims": True,
                "unsupported_concrete_claims": True,
                "covers_requested_facets": False,
                "requires_abstention": False,
                "pass_postcondition": False,
                "failure_reasons": ["unsupported_concrete_claim"],
                "explanation": "The retry still adds downtown.",
            },
            {
                "readable": True,
                "is_abstention": False,
                "contains_concrete_claims": True,
                "unsupported_concrete_claims": True,
                "covers_requested_facets": False,
                "requires_abstention": False,
                "pass_postcondition": False,
                "failure_reasons": ["unsupported_concrete_claim"],
                "explanation": "The support repair still adds downtown.",
            },
        ],
    )

    result = await complete_answer_with_postcondition_guard(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        request=_request(),
        verifier_model="openai/gpt-5-mini",
        original_query="Which cities did Caroline mention?",
        composed_context=_answer_support_context(),
        privacy_enforcement="off",
    )

    assert result.output_text == "Paris, Rome"
    assert result.report.status == "supported_partial_fallback"
    assert result.report.abstention_reason == "supported_answer_repair_failed"
    assert result.report.evidence_use_repair_count == 1
    assert any(
        request.metadata.get("atagia_answer_evidence_use_repair")
        for request in provider.requests
    )


@pytest.mark.parametrize(
    ("failure_reasons", "abstention_reason"),
    [
        (["verifier_failed"], "verifier_failed"),
        (["verifier_structured_output_failure"], "verifier_structured_output_failure"),
        (["empty_output"], "empty_output"),
        (["unreadable_output"], "answer_postcondition_failed"),
        (["unsupported_concrete_claim"], "answer_postcondition_failed"),
        (["incomplete_requested_facets"], "answer_postcondition_failed"),
        (["output_limit_exceeded"], "answer_postcondition_failed"),
        (["answer_evidence_use_failure"], "answer_postcondition_failed"),
        (["missing_required_abstention"], "supported_answer_repair_failed"),
    ],
)
def test_supported_partial_fallback_requires_supported_repair_abstention_reason(
    failure_reasons: list[str],
    abstention_reason: str,
) -> None:
    fallback_text = "I do not have enough supported evidence to answer."
    result = _abstain_result(
        request=_request(),
        response=LLMCompletionResponse(
            provider="answer-guard-tests",
            model="openai/gpt-5-mini",
            output_text="Caroline mentioned Paris, Rome, and downtown.",
        ),
        composed_context=_answer_support_context(),
        retry_count=0,
        output_limit_retry=False,
        initial_output_chars=45,
        failure_reasons=failure_reasons,
        fallback_text=fallback_text,
        abstention_reason=abstention_reason,
        repair_verdict_existed=True,
    )

    assert result.output_text == fallback_text
    assert result.report.status == "abstained"
    assert result.report.abstention_reason == abstention_reason
    assert "retrieved evidence supports" not in result.output_text


def test_supported_partial_fallback_requires_repair_verdict() -> None:
    fallback_text = "I do not have enough supported evidence to answer."
    result = _abstain_result(
        request=_request(),
        response=LLMCompletionResponse(
            provider="answer-guard-tests",
            model="openai/gpt-5-mini",
            output_text="",
        ),
        composed_context=_answer_support_context(),
        retry_count=1,
        output_limit_retry=False,
        initial_output_chars=45,
        failure_reasons=["answer_evidence_use_failure"],
        fallback_text=fallback_text,
        abstention_reason="supported_answer_repair_failed",
        repair_verdict_existed=False,
    )

    assert result.output_text == fallback_text
    assert result.report.status == "abstained"
    assert result.report.abstention_reason == "supported_answer_repair_failed"


@pytest.mark.asyncio
async def test_answer_postcondition_repair_includes_multiple_broad_list_evidence_items() -> (
    None
):
    provider = AnswerGuardProvider(
        outputs=[
            "Jon has visited Rome.",
            "Jon has visited Paris and Rome.",
        ],
        verdicts=[
            _pass_verdict(),
            {
                "uses_required_evidence": False,
                "should_repair": True,
                "reason": "The answer omitted a direct list item.",
                "missing_supported_obligations": [
                    "The claim that Jon visited Paris is unsupported."
                ],
                "evidence_ids_supporting_answer": ["sum_rome", "vew_paris"],
            },
            _pass_verdict(),
        ],
    )

    result = await complete_answer_with_postcondition_guard(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        request=_request(),
        verifier_model="openai/gpt-5-mini",
        original_query="Which cities has Jon visited?",
        composed_context=_answer_evidence_context(),
        retrieval_sufficiency={
            "state": "retrieval_sufficient",
            "would_abstain": False,
        },
        retrieval_diagnostics=_broad_list_answer_evidence_retrieval_diagnostics(),
        privacy_enforcement="off",
    )

    assert result.output_text == "Jon has visited Paris and Rome."
    repair_request = next(
        request
        for request in provider.requests
        if request.metadata.get("atagia_answer_evidence_use_repair")
    )
    repair_payload = "; ".join(
        repair_request.metadata["atagia_answer_evidence_use_repair_obligations"]
    )
    assert "Rome" in repair_payload
    assert "Paris" in repair_payload


@pytest.mark.asyncio
async def test_answer_postcondition_reactive_verifier_distinguishes_category_from_detail() -> None:
    provider = AnswerGuardProvider(
        outputs=["No allergies are documented, but related medical context may exist."],
        verdicts=[_pass_verdict()],
    )
    result = await complete_answer_with_postcondition_guard(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        request=_request(),
        verifier_model="openai/gpt-5-mini",
        original_query="Does Rosa have any allergies?",
        composed_context=_context(),
        answer_stance="reactive",
    )

    assert result.report.status == "passed"
    verifier_request = next(
        request
        for request in provider.requests
        if request.metadata.get("purpose") == "answer_postcondition_verification"
    )
    assert "answer_stance=reactive" in verifier_request.messages[1].content
    assert "medication name" in verifier_request.messages[1].content
    assert "broad category signal" in verifier_request.messages[1].content


@pytest.mark.asyncio
async def test_answer_postcondition_privacy_off_ignores_policy_only_abstention() -> (
    None
):
    provider = AnswerGuardProvider(
        outputs=["The production jump host password is K8sN0d3Jump!2024."],
        verdicts=[
            {
                **_pass_verdict(),
                "requires_abstention": True,
                "pass_postcondition": False,
                "failure_reasons": ["missing_required_abstention"],
                "explanation": "The answer reveals a credential and should abstain.",
            }
        ],
    )

    result = await complete_answer_with_postcondition_guard(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        request=_request(),
        verifier_model="openai/gpt-5-mini",
        original_query="What is the production jump host password?",
        composed_context=_secret_context(),
        retrieval_sufficiency={
            "state": "retrieval_sufficient",
            "would_abstain": False,
        },
        privacy_enforcement="off",
    )

    assert (
        result.output_text == "The production jump host password is K8sN0d3Jump!2024."
    )
    assert result.report.status == "passed"
    assert result.report.retry_count == 0
    assert result.report.failure_reasons == []
    assert result.report.verdict is not None
    assert result.report.verdict.requires_abstention is False
    assert result.report.verdict.pass_postcondition is True
    assert result.report.verdict.failure_reasons == []
    verifier_request = next(
        request
        for request in provider.requests
        if request.metadata.get("purpose") == "answer_postcondition_verification"
    )
    assert "Privacy restrictions are inactive" in verifier_request.messages[0].content
    assert (
        "Privacy restrictions are inactive for this request"
        in verifier_request.messages[1].content
    )
    assert "historical source-time content" in verifier_request.messages[1].content
    assert "original query is part of the request context" in (
        verifier_request.messages[1].content
    )


@pytest.mark.asyncio
async def test_answer_postcondition_retries_unreadable_answer() -> None:
    provider = AnswerGuardProvider(
        outputs=["}?? garbled", "Rosa mentioned naproxen 500 mg twice a day."],
        verdicts=[
            {
                **_pass_verdict(),
                "readable": False,
                "pass_postcondition": False,
                "failure_reasons": ["unreadable_output"],
            },
            _pass_verdict(),
        ],
    )

    result = await complete_answer_with_postcondition_guard(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        request=_request(),
        verifier_model="openai/gpt-5-mini",
        original_query="What medication did Rosa mention?",
        composed_context=_context(),
    )

    assert result.output_text == "Rosa mentioned naproxen 500 mg twice a day."
    assert result.report.status == "retry_passed"
    assert result.report.retry_count == 1
    retry_request = next(
        request
        for request in provider.requests
        if request.metadata.get("atagia_answer_postcondition_retry")
    )
    assert "unreadable_output" in retry_request.messages[0].content


@pytest.mark.asyncio
async def test_answer_postcondition_normalizes_free_form_failure_reasons() -> None:
    provider = AnswerGuardProvider(
        outputs=["Rosa took aspirin.", "Rosa mentioned naproxen 500 mg twice a day."],
        verdicts=[
            {
                **_pass_verdict(),
                "unsupported_concrete_claims": True,
                "pass_postcondition": False,
                "failure_reasons": ["not_supported_by_context"],
            },
            _pass_verdict(),
        ],
    )

    result = await complete_answer_with_postcondition_guard(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        request=_request(),
        verifier_model="openai/gpt-5-mini",
        original_query="What medication did Rosa mention?",
        composed_context=_context(),
    )

    assert result.output_text == "Rosa mentioned naproxen 500 mg twice a day."
    retry_request = next(
        request
        for request in provider.requests
        if request.metadata.get("atagia_answer_postcondition_retry")
    )
    assert "unsupported_concrete_claim" in retry_request.messages[0].content


@pytest.mark.asyncio
async def test_answer_postcondition_retry_includes_claim_pruning_diagnostics() -> None:
    provider = AnswerGuardProvider(
        outputs=[
            "Rosa mentioned naproxen 500 mg twice a day and also started aspirin.",
            "Rosa mentioned naproxen 500 mg twice a day.",
        ],
        verdicts=[
            {
                **_pass_verdict(),
                "unsupported_concrete_claims": True,
                "pass_postcondition": False,
                "failure_reasons": ["unsupported_concrete_claim"],
                "explanation": "Aspirin is not supported by retrieved evidence.",
            },
            _pass_verdict(),
        ],
    )

    result = await complete_answer_with_postcondition_guard(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        request=_request(),
        verifier_model="openai/gpt-5-mini",
        original_query="What medication did Rosa mention?",
        composed_context=_context(),
    )

    assert result.output_text == "Rosa mentioned naproxen 500 mg twice a day."
    retry_request = next(
        request
        for request in provider.requests
        if request.metadata.get("atagia_answer_postcondition_retry")
    )
    retry_prompt = retry_request.messages[0].content
    assert "Answer repair diagnostics follow. They are not evidence." in retry_prompt
    assert "preserve supported requested facets" in retry_prompt
    assert "Aspirin is not supported by retrieved evidence." in retry_prompt
    assert "previous_answer" in retry_prompt


@pytest.mark.asyncio
async def test_answer_postcondition_normalizes_unknown_bool_values() -> None:
    provider = AnswerGuardProvider(
        outputs=["Rosa took aspirin.", "Rosa mentioned naproxen 500 mg twice a day."],
        verdicts=[
            {
                **_pass_verdict(),
                "unsupported_concrete_claims": "unknown",
                "pass_postcondition": False,
            },
            _pass_verdict(),
        ],
    )

    result = await complete_answer_with_postcondition_guard(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        request=_request(),
        verifier_model="openai/gpt-5-mini",
        original_query="What medication did Rosa mention?",
        composed_context=_context(),
    )

    assert result.output_text == "Rosa mentioned naproxen 500 mg twice a day."
    retry_request = next(
        request
        for request in provider.requests
        if request.metadata.get("atagia_answer_postcondition_retry")
    )
    assert "unsupported_concrete_claim" in retry_request.messages[0].content


@pytest.mark.asyncio
async def test_answer_postcondition_accepts_explicit_pass_with_empty_failure_labels() -> (
    None
):
    provider = AnswerGuardProvider(
        outputs=["Rosa mentioned naproxen 500 mg twice a day."],
        verdicts=[
            {
                **_pass_verdict(),
                "unsupported_concrete_claims": "unknown",
            }
        ],
    )

    result = await complete_answer_with_postcondition_guard(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        request=_request(),
        verifier_model="openai/gpt-5-mini",
        original_query="What medication did Rosa mention?",
        composed_context=_context(),
    )

    assert result.output_text == "Rosa mentioned naproxen 500 mg twice a day."
    assert result.report.status == "passed"


@pytest.mark.asyncio
async def test_answer_postcondition_retries_output_limit() -> None:
    provider = AnswerGuardProvider(
        outputs=["I do not have enough reliable retrieved evidence."],
        verdicts=[
            {
                **_pass_verdict(),
                "is_abstention": True,
                "contains_concrete_claims": False,
            }
        ],
    )
    provider.raise_output_limit_once = True

    result = await complete_answer_with_postcondition_guard(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        request=_request(),
        verifier_model="openai/gpt-5-mini",
        original_query="What medication did Rosa mention?",
        composed_context=_context(),
    )

    assert result.output_text == "I do not have enough reliable retrieved evidence."
    assert result.report.status == "passed"
    assert result.report.output_limit_retry is False


@pytest.mark.asyncio
async def test_answer_postcondition_abstains_when_retry_hits_output_limit() -> None:
    failing_verdict = {
        **_pass_verdict(),
        "unsupported_concrete_claims": True,
        "pass_postcondition": False,
        "failure_reasons": ["unsupported_concrete_claim"],
    }
    provider = AnswerGuardProvider(
        outputs=["unsupported fact"],
        verdicts=[failing_verdict],
    )
    provider.raise_output_limit_on_retry_once = True

    result = await complete_answer_with_postcondition_guard(
        llm_client=LLMClient(
            provider_name=provider.name,
            providers=[provider],
            technical_recovery_config=LLMTechnicalRecoveryConfig.disabled(),
        ),
        request=_request(),
        verifier_model="openai/gpt-5-mini",
        original_query="What medication did Rosa mention?",
        composed_context=_context(),
    )

    assert result.output_text == (
        "I do not have enough reliable retrieved evidence to answer that safely."
    )
    assert result.report.status == "abstained"
    assert result.report.failure_reasons == ["empty_output"]
    assert result.report.abstention_reason == "empty_output"
    assert result.report.retry_count == 1
    assert any(
        request.metadata.get("atagia_answer_postcondition_retry")
        for request in provider.requests
    )


@pytest.mark.asyncio
async def test_answer_postcondition_abstains_after_failed_retry() -> None:
    failing_verdict = {
        **_pass_verdict(),
        "unsupported_concrete_claims": True,
        "pass_postcondition": False,
        "failure_reasons": ["unsupported_concrete_claim"],
    }
    provider = AnswerGuardProvider(
        outputs=["unsupported fact", "still unsupported"],
        verdicts=[failing_verdict, failing_verdict],
    )

    result = await complete_answer_with_postcondition_guard(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        request=_request(),
        verifier_model="openai/gpt-5-mini",
        original_query="What medication did Rosa mention?",
        composed_context=_context(),
    )

    assert result.output_text == (
        "I do not have enough reliable retrieved evidence to answer that safely."
    )
    assert result.report.status == "abstained"
    assert result.report.failure_reasons == ["unsupported_concrete_claim"]
    assert result.report.abstention_reason == "answer_postcondition_failed"


@pytest.mark.asyncio
async def test_answer_postcondition_repairs_invalid_verifier_output() -> None:
    provider = AnswerGuardProvider(
        outputs=["Rosa mentioned naproxen 500 mg twice a day."],
        verdicts=["not json", _pass_verdict()],
    )
    result = await complete_answer_with_postcondition_guard(
        llm_client=LLMClient(
            provider_name=provider.name,
            providers=[provider],
            structured_output_retry_attempts=0,
        ),
        request=_request(),
        verifier_model="openai/gpt-5-mini",
        original_query="What medication did Rosa mention?",
        composed_context=_context(),
    )

    assert result.output_text == "Rosa mentioned naproxen 500 mg twice a day."
    assert result.report.status == "passed"
    assert result.report.verifier_retry_count == 1
    assert result.report.verifier_structured_output_retry_count == 1
    assert result.report.verifier_structured_output_retry_success_count == 1
    assert result.report.verifier_structured_output_failure_count == 0
    verifier_retry_request = next(
        request
        for request in provider.requests
        if request.metadata.get("atagia_answer_postcondition_verifier_retry")
    )
    assert verifier_retry_request.max_output_tokens == 8192
    assert (
        "Return only the complete JSON object"
        in verifier_retry_request.messages[-1].content
    )


@pytest.mark.asyncio
async def test_answer_postcondition_fails_closed_after_invalid_verifier_retry() -> None:
    provider = AnswerGuardProvider(
        outputs=["Rosa mentioned naproxen 500 mg twice a day."],
        verdicts=["not json", "still not json"],
    )
    result = await complete_answer_with_postcondition_guard(
        llm_client=LLMClient(
            provider_name=provider.name,
            providers=[provider],
            structured_output_retry_attempts=0,
        ),
        request=_request(),
        verifier_model="openai/gpt-5-mini",
        original_query="What medication did Rosa mention?",
        composed_context=_context(),
    )

    assert result.output_text == (
        "I do not have enough reliable retrieved evidence to answer that safely."
    )
    assert result.report.status == "abstained"
    assert result.report.failure_reasons == ["verifier_structured_output_failure"]
    assert result.report.abstention_reason == "verifier_structured_output_failure"
    assert result.report.verifier_retry_count == 1
    assert result.report.verifier_structured_output_retry_count == 1
    assert result.report.verifier_structured_output_retry_success_count == 0
    assert result.report.verifier_structured_output_failure_count == 1
    assert result.report.verifier_error_class == "StructuredOutputError"


@pytest.mark.asyncio
async def test_answer_postcondition_repairs_illegitimate_abstention() -> None:
    provider = AnswerGuardProvider(
        outputs=[
            "I do not have enough reliable retrieved evidence.",
            "Rosa mentioned naproxen 500 mg twice a day.",
        ],
        verdicts=[
            {
                **_pass_verdict(),
                "is_abstention": True,
                "contains_concrete_claims": False,
                "covers_requested_facets": False,
                "requires_abstention": True,
            },
            {
                "abstention_allowed": False,
                "reason": "The selected memory directly answers the medication question.",
                "missing_supported_obligations": ["What medication did Rosa mention?"],
                "evidence_ids_supporting_answer": ["mem_1"],
                "policy_or_scope_blocker": False,
                "evidence_insufficient": False,
            },
            _pass_verdict(),
        ],
    )

    result = await complete_answer_with_postcondition_guard(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        request=_request(),
        verifier_model="openai/gpt-5-mini",
        original_query="What medication did Rosa mention?",
        composed_context=_context(),
        retrieval_sufficiency={
            "state": "retrieval_sufficient",
            "would_abstain": False,
        },
        retrieval_diagnostics=_supported_retrieval_diagnostics(),
    )

    assert result.output_text == "Rosa mentioned naproxen 500 mg twice a day."
    assert result.report.status == "retry_passed"
    assert result.report.evidence_use_repair_count == 1
    assert result.report.evidence_use_repair_success_count == 1
    assert result.report.supported_abstention_detected is True
    assert result.report.final_answer_used_required_evidence is True
    assert {event.mechanism for event in result.report.temporary_scaffolding} == {
        "abstention_legitimacy_review",
        "evidence_use_repair",
        "answer_retry",
    }
    repair_request = next(
        request
        for request in provider.requests
        if request.metadata.get("atagia_answer_evidence_use_repair")
    )
    assert repair_request.max_output_tokens == 8192
    assert "Put the current requested value first" in repair_request.messages[0].content
    assert (
        "omit them unless the user asked for history or comparison"
        in repair_request.messages[0].content
    )


@pytest.mark.asyncio
async def test_answer_postcondition_repairs_abstention_when_legitimacy_has_evidence_ids_only() -> (
    None
):
    provider = AnswerGuardProvider(
        outputs=[
            "I do not have enough reliable retrieved evidence.",
            "Rosa mentioned naproxen 500 mg twice a day.",
        ],
        verdicts=[
            {
                **_pass_verdict(),
                "is_abstention": True,
                "contains_concrete_claims": False,
                "covers_requested_facets": False,
                "requires_abstention": True,
            },
            {
                "abstention_allowed": False,
                "reason": (
                    "The selected memory directly answers the medication question."
                ),
                "missing_supported_obligations": [],
                "evidence_ids_supporting_answer": ["mem_1"],
                "policy_or_scope_blocker": False,
                "evidence_insufficient": False,
            },
            _pass_verdict(),
        ],
    )

    result = await complete_answer_with_postcondition_guard(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        request=_request(),
        verifier_model="openai/gpt-5-mini",
        original_query="What medication did Rosa mention?",
        composed_context=_context(),
        retrieval_sufficiency={
            "state": "retrieval_sufficient",
            "would_abstain": False,
        },
        retrieval_diagnostics=_supported_retrieval_diagnostics(),
    )

    assert result.output_text == "Rosa mentioned naproxen 500 mg twice a day."
    assert result.report.status == "retry_passed"
    assert result.report.evidence_use_repair_count == 1
    assert result.report.evidence_use_repair_success_count == 1
    assert result.report.supported_abstention_detected is True
    assert result.report.missing_supported_obligations == []
    repair_request = next(
        request
        for request in provider.requests
        if request.metadata.get("atagia_answer_evidence_use_repair")
    )
    assert "What medication did Rosa mention?" in repair_request.messages[0].content
    assert (
        repair_request.metadata["atagia_answer_evidence_use_repair_obligations"]
        == ["What medication did Rosa mention?"]
    )


@pytest.mark.asyncio
async def test_answer_postcondition_repairs_abstention_from_answer_evidence() -> None:
    provider = AnswerGuardProvider(
        outputs=[
            "I do not have enough reliable retrieved evidence.",
            "Jon wanted to savor all the good vibes.",
        ],
        verdicts=[
            {
                **_pass_verdict(),
                "is_abstention": True,
                "contains_concrete_claims": False,
                "covers_requested_facets": False,
                "requires_abstention": True,
            },
            {
                "abstention_allowed": False,
                "reason": "The answer evidence contains a direct source quote.",
                "missing_supported_obligations": [],
                "evidence_ids_supporting_answer": ["mem_vibes"],
                "policy_or_scope_blocker": False,
                "evidence_insufficient": False,
            },
            _pass_verdict(),
        ],
    )

    result = await complete_answer_with_postcondition_guard(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        request=_request(),
        verifier_model="openai/gpt-5-mini",
        original_query="What did Jon want to savor?",
        composed_context=_answer_evidence_context(),
        retrieval_sufficiency={
            "state": "retrieval_sufficient",
            "would_abstain": False,
        },
        retrieval_diagnostics=_answer_evidence_retrieval_diagnostics(),
        privacy_enforcement="off",
    )

    assert result.output_text == "Jon wanted to savor all the good vibes."
    assert result.report.status == "retry_passed"
    assert result.report.evidence_use_repair_count == 1
    repair_request = next(
        request
        for request in provider.requests
        if request.metadata.get("atagia_answer_evidence_use_repair")
    )
    repair_payload = "; ".join(
        repair_request.metadata["atagia_answer_evidence_use_repair_obligations"]
    )
    assert "mem_vibes" in repair_payload
    assert "Jon: I want to savor all the good vibes." in repair_payload


@pytest.mark.asyncio
async def test_answer_postcondition_repairs_abstention_with_plan_diagnostics() -> None:
    provider = AnswerGuardProvider(
        outputs=[
            "I do not have enough reliable retrieved evidence.",
            "Rosa mentioned naproxen 500 mg twice a day.",
        ],
        verdicts=[
            {
                **_pass_verdict(),
                "is_abstention": True,
                "contains_concrete_claims": False,
                "covers_requested_facets": False,
                "requires_abstention": True,
            },
            {
                "abstention_allowed": False,
                "reason": "The selected memory directly answers the medication question.",
                "missing_supported_obligations": ["What medication did Rosa mention?"],
                "evidence_ids_supporting_answer": ["mem_1"],
                "policy_or_scope_blocker": False,
                "evidence_insufficient": False,
            },
            _pass_verdict(),
        ],
    )

    result = await complete_answer_with_postcondition_guard(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        request=_request(),
        verifier_model="openai/gpt-5-mini",
        original_query="What medication did Rosa mention?",
        composed_context=_context(),
        retrieval_sufficiency={
            "state": "retrieval_sufficient",
            "would_abstain": False,
        },
        retrieval_diagnostics=_supported_retrieval_plan_diagnostics(),
    )

    assert result.output_text == "Rosa mentioned naproxen 500 mg twice a day."
    assert result.report.status == "retry_passed"
    assert result.report.evidence_use_repair_count == 1
    assert result.report.abstention_legitimacy_verdict is not None
    legitimacy_request = next(
        request
        for request in provider.requests
        if request.metadata.get("purpose")
        == "answer_abstention_legitimacy_verification"
    )
    prompt = legitimacy_request.messages[1].content
    assert '"retrieval_plan"' in prompt
    assert '"diagnostic_shape_fallback_used": true' in prompt


@pytest.mark.asyncio
async def test_answer_postcondition_repairs_failed_retry_with_supported_evidence() -> (
    None
):
    provider = AnswerGuardProvider(
        outputs=[
            "Rosa mentioned naproxen 500 mg twice a day and also started aspirin.",
            "Rosa started aspirin.",
            "Rosa mentioned naproxen 500 mg twice a day.",
        ],
        verdicts=[
            {
                **_pass_verdict(),
                "unsupported_concrete_claims": True,
                "pass_postcondition": False,
                "failure_reasons": ["unsupported_concrete_claim"],
            },
            {
                **_pass_verdict(),
                "unsupported_concrete_claims": True,
                "pass_postcondition": False,
                "failure_reasons": ["unsupported_concrete_claim"],
            },
            _pass_verdict(),
        ],
    )

    result = await complete_answer_with_postcondition_guard(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        request=_request(),
        verifier_model="openai/gpt-5-mini",
        original_query="What medication did Rosa mention?",
        composed_context=_context(),
        retrieval_sufficiency={
            "state": "retrieval_sufficient",
            "would_abstain": False,
        },
        retrieval_diagnostics=_supported_retrieval_diagnostics(),
    )

    assert result.output_text == "Rosa mentioned naproxen 500 mg twice a day."
    assert result.report.status == "retry_passed"
    assert result.report.retry_count == 2
    assert result.report.evidence_use_repair_count == 1
    assert result.report.evidence_use_repair_success_count == 1
    assert result.report.supported_abstention_detected is True
    assert "evidence_use_repair" in {
        event.mechanism for event in result.report.temporary_scaffolding
    }
    assert any(
        request.metadata.get("atagia_answer_postcondition_retry")
        for request in provider.requests
    )
    repair_request = next(
        request
        for request in provider.requests
        if request.metadata.get("atagia_answer_evidence_use_repair")
    )
    assert (
        "failed to use selected eligible evidence" in repair_request.messages[0].content
    )
    assert not any(
        request.metadata.get("purpose") == "answer_abstention_legitimacy_verification"
        for request in provider.requests
    )


@pytest.mark.asyncio
async def test_answer_postcondition_privacy_off_disables_policy_abstention_blockers() -> (
    None
):
    provider = AnswerGuardProvider(
        outputs=[
            "I cannot reveal that.",
            "Rosa mentioned naproxen 500 mg twice a day.",
        ],
        verdicts=[
            {
                **_pass_verdict(),
                "is_abstention": True,
                "contains_concrete_claims": False,
                "covers_requested_facets": False,
                "requires_abstention": True,
            },
            {
                "abstention_allowed": False,
                "reason": "Privacy is off and the selected memory answers the question.",
                "missing_supported_obligations": ["What medication did Rosa mention?"],
                "evidence_ids_supporting_answer": ["mem_1"],
                "policy_or_scope_blocker": False,
                "evidence_insufficient": False,
            },
            _pass_verdict(),
        ],
    )

    result = await complete_answer_with_postcondition_guard(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        request=_request(),
        verifier_model="openai/gpt-5-mini",
        original_query="What medication did Rosa mention?",
        composed_context=_context(),
        retrieval_sufficiency={
            "state": "retrieval_sufficient",
            "would_abstain": False,
        },
        retrieval_diagnostics=_supported_retrieval_diagnostics(),
        privacy_enforcement="off",
    )

    assert result.output_text == "Rosa mentioned naproxen 500 mg twice a day."
    legitimacy_request = next(
        request
        for request in provider.requests
        if request.metadata.get("purpose")
        == "answer_abstention_legitimacy_verification"
    )
    assert (
        "With privacy restrictions inactive" in legitimacy_request.messages[0].content
    )
    assert (
        "Privacy restrictions are inactive for this request."
        in legitimacy_request.messages[1].content
    )
    assert (
        "historical evidence, not active blockers"
        in legitimacy_request.messages[1].content
    )


@pytest.mark.asyncio
async def test_answer_postcondition_master_repairs_source_time_privacy_abstention() -> (
    None
):
    provider = AnswerGuardProvider(
        outputs=[
            "I cannot share that because Ben asked to keep it private.",
            "Ben is seeing Dr. Reeves for anxiety.",
        ],
        verdicts=[
            {
                **_pass_verdict(),
                "is_abstention": True,
                "contains_concrete_claims": False,
                "covers_requested_facets": False,
                "requires_abstention": True,
            },
            {
                "abstention_allowed": False,
                "reason": (
                    "The selected base evidence answers the question; the "
                    "source-time privacy request is not active authority here."
                ),
                "missing_supported_obligations": [
                    "What private therapy detail did Ben share?"
                ],
                "evidence_ids_supporting_answer": ["mem_1"],
                "policy_or_scope_blocker": False,
                "evidence_insufficient": False,
            },
            _pass_verdict(),
        ],
    )

    result = await complete_answer_with_postcondition_guard(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        request=_request(),
        verifier_model="openai/gpt-5-mini",
        original_query="What private therapy detail did Ben share?",
        composed_context=_source_time_private_context(),
        retrieval_sufficiency={
            "state": "retrieval_sufficient",
            "would_abstain": False,
        },
        retrieval_diagnostics=_supported_retrieval_diagnostics(),
        privacy_enforcement="enforce",
        prompt_authority_context=normalize_request_authority_context(
            privacy_enforcement="enforce",
            authenticated_user_privilege_level="atagia_master",
            authenticated_user_is_atagia_master=True,
        ),
    )

    assert result.output_text == "Ben is seeing Dr. Reeves for anxiety."
    assert result.report.status == "retry_passed"
    assert result.report.evidence_use_repair_count == 1
    verifier_request = next(
        request
        for request in provider.requests
        if request.metadata.get("purpose") == "answer_postcondition_verification"
    )
    assert "Privacy restrictions are inactive" in verifier_request.messages[0].content
    assert (
        "source-time privacy/disclosure statements"
        in verifier_request.messages[0].content
    )
    assert "original query is part of the request context" in (
        verifier_request.messages[1].content
    )


@pytest.mark.asyncio
async def test_answer_postcondition_repairs_failed_illegitimate_abstention() -> None:
    provider = AnswerGuardProvider(
        outputs=[
            "I cannot answer that safely.",
            "Rosa mentioned naproxen 500 mg twice a day.",
        ],
        verdicts=[
            {
                **_pass_verdict(),
                "is_abstention": True,
                "contains_concrete_claims": False,
                "covers_requested_facets": False,
                "requires_abstention": False,
                "pass_postcondition": False,
                "failure_reasons": ["verifier_failed"],
            },
            {
                "abstention_allowed": False,
                "reason": "The selected memory directly answers the medication question.",
                "missing_supported_obligations": ["What medication did Rosa mention?"],
                "evidence_ids_supporting_answer": ["mem_1"],
                "policy_or_scope_blocker": False,
                "evidence_insufficient": False,
            },
            _pass_verdict(),
        ],
    )

    result = await complete_answer_with_postcondition_guard(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        request=_request(),
        verifier_model="openai/gpt-5-mini",
        original_query="What medication did Rosa mention?",
        composed_context=_context(),
        retrieval_sufficiency={
            "state": "retrieval_sufficient",
            "would_abstain": False,
        },
        retrieval_diagnostics=_supported_retrieval_diagnostics(),
    )

    assert result.output_text == "Rosa mentioned naproxen 500 mg twice a day."
    assert result.report.status == "retry_passed"
    assert result.report.evidence_use_repair_count == 1
    assert result.report.evidence_use_repair_success_count == 1
    assert result.report.supported_abstention_detected is True
    assert not any(
        request.metadata.get("atagia_answer_postcondition_retry")
        for request in provider.requests
    )
    assert any(
        request.metadata.get("atagia_answer_evidence_use_repair")
        for request in provider.requests
    )


@pytest.mark.asyncio
async def test_answer_postcondition_keeps_legitimate_abstention() -> None:
    provider = AnswerGuardProvider(
        outputs=["I do not have enough reliable retrieved evidence."],
        verdicts=[
            {
                **_pass_verdict(),
                "is_abstention": True,
                "contains_concrete_claims": False,
                "covers_requested_facets": False,
                "requires_abstention": True,
            },
            {
                "abstention_allowed": True,
                "reason": "The retrieved context does not answer the question.",
                "missing_supported_obligations": [],
                "evidence_ids_supporting_answer": [],
                "policy_or_scope_blocker": False,
                "evidence_insufficient": True,
            },
        ],
    )

    result = await complete_answer_with_postcondition_guard(
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        request=_request(),
        verifier_model="openai/gpt-5-mini",
        original_query="What medication did Rosa mention?",
        composed_context=_context(),
        retrieval_sufficiency={
            "state": "retrieval_sufficient",
            "would_abstain": False,
        },
        retrieval_diagnostics=_supported_retrieval_diagnostics(),
    )

    assert result.output_text == "I do not have enough reliable retrieved evidence."
    assert result.report.status == "passed"
    assert result.report.evidence_use_repair_count == 0
    assert result.report.supported_abstention_detected is False
    assert result.report.abstention_legitimacy_verdict is not None
    assert result.report.abstention_legitimacy_verdict.abstention_allowed is True
