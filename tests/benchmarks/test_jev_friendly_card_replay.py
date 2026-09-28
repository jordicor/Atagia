"""Offline route checks for the seven synthetic card replay families."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from atagia.core.source_references import SourceReferenceCatalog
from atagia.models.schemas_decisions import ChoiceAnswer, ScoreAnswer
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)
from benchmarks.jev_friendly_cards.card_replay import run_card_replay


ROOT = Path(__file__).resolve().parents[2]
MODEL = "openrouter/openai/gpt-6-luna"
SOURCE = "Ali and Nora will update the launch checklist tomorrow."


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        manifests_path=str(ROOT / "src" / "atagia" / "resources" / "manifests"),
        llm_component_models={
            component: MODEL
            for component in (
                "extractor",
                "extraction_evidence",
                "extraction_kind",
                "extraction_scope",
                "extraction_confidence",
                "extraction_evidence_support",
                "extraction_preserve_verbatim",
                "extraction_temporal_type",
                "extraction_member_identity",
                "need_detector_query_language",
                "need_detector_answer_language",
                "intent_classifier",
                "topic_working_set",
            )
        },
        llm_component_examples={},
        card_examples_enabled=False,
        topic_working_set_update_mode="direct",
        llm_finite_decisions_enabled=False,
    )


def _case(family: str) -> dict:
    return {
        "case_id": "S01",
        "origin": "synthetic",
        "primary_family": family,
        "source_text": SOURCE,
        "role": "user",
        "context_messages": [],
    }


def _fixture() -> dict:
    return {
        "user_id": "eval_S01",
        "conversation_id": "eval_S01",
        "source_message_id": "S01",
        "assistant_mode_id": "general_qa",
        "mode": "general_qa",
        "privacy_enforcement": "off",
        "occurred_at": "2026-09-26T12:00:00+00:00",
        "allowed_write_scopes": ["chat", "user"],
        "prior_chunk_context": "The team discussed the checklist yesterday.",
        "card_replay_candidate": "Ali and Nora will update the checklist tomorrow.",
        "belief": {
            "candidate_key": "projects.launch.priority",
            "catalog_keys": ["projects.launch.ownership"],
        },
        "member_identity": {"extracted_member_labels": ["Ali", "Nora"]},
        "topic": {
            "snapshot": {
                "active_topics": [
                    {
                        "id": "tpc_launch",
                        "title": "Launch checklist",
                        "summary": "The team tracks the launch.",
                        "active_goal": "Finish the checklist.",
                        "open_questions": [],
                        "decisions": [],
                    }
                ],
                "parked_topics": [],
            }
        },
    }


class ReplayProvider(LLMProvider):
    name = "openrouter"

    def __init__(self) -> None:
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        purpose = request.metadata["purpose"]
        last_ref = SourceReferenceCatalog(SOURCE).anchors[-1].reference_id
        outputs = {
            "memory_extraction_kind_card": "evidence",
            "memory_extraction_scope_card": "user",
            "memory_extraction_confidence_card": "0.85",
            "memory_extraction_evidence_support_card": "direct",
            "memory_extraction_preserve_verbatim_card": "yes",
            "memory_extraction_candidate_language_card": "en",
            "memory_extraction_source_reference_card": f"r1 {last_ref}",
            "memory_extraction_temporal_type_card": "bounded",
            "memory_extraction_temporal_interval_card": (
                "2026-09-26T12:00:00+00:00\n2026-09-27T12:00:00+00:00"
            ),
            "need_detection_query_language_card": "en",
            "need_detection_answer_language_card": "en",
            "memory_extraction_belief_key_card": "projects.launch.deadline",
            "memory_extraction_belief_value_card": "tomorrow",
            "intent_classifier_claim_key_equivalence_batch": "no",
            "memory_extraction_coverage_members_card": '"Ali"\n"Nora"',
            "memory_extraction_coverage_member_identity_card": "self",
            "topic_working_set_route_card": "update tpc_launch S01",
            "topic_working_set_title_card": "none",
            "topic_working_set_summary_card": "The team will update the launch checklist.",
            "topic_working_set_goal_card": "none",
            "topic_working_set_questions_card": "none",
            "topic_working_set_decisions_card": "none",
            "topic_working_set_boundary_card": "tpc_launch ordinary 0 0.8",
        }
        if purpose not in outputs:
            raise AssertionError(f"Unexpected production call: {purpose}")
        output = (
            '{"results":[{"pair_id":"pair_0","equivalent":false}]}'
            if purpose == "intent_classifier_claim_key_equivalence_batch"
            and request.response_schema is not None
            else outputs[purpose]
        )
        if purpose == "memory_extraction_coverage_member_identity_card":
            prompt = request.messages[-1].content
            if "<member>" in prompt:
                output = "Ali" if "<member>\nAli\n" in prompt else "Nora"
        return LLMCompletionResponse(
            provider=self.name, model=request.model, output_text=output
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Card replay must not embed")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "family",
    [
        "classification",
        "evidence",
        "temporal",
        "language",
        "beliefs",
        "members",
        "topics",
    ],
)
async def test_card_replay_uses_production_routes_and_serializes(family: str) -> None:
    provider = ReplayProvider()
    client = LLMClient(providers=[provider], structured_output_retry_attempts=0)
    result = await run_card_replay(
        client, case=_case(family), fixture=_fixture(), settings=_settings()
    )

    assert json.loads(json.dumps(result)) == result
    assert result["case_id"] == "S01"
    assert result["family"] == family
    assert result["candidate_id"] == "cand_001"
    purposes = [request.metadata["purpose"] for request in provider.requests]
    assert purposes
    assert all(
        request.model == MODEL.removeprefix("openrouter/")
        for request in provider.requests
    )

    if family == "classification":
        assert set(result["cards"]) == {
            "memory_kind",
            "memory_scope",
            "memory_confidence",
        }
        assert result["cards"]["memory_confidence"]["parsed"] == {"cand_001": 0.85}
        assert purposes == [
            "memory_extraction_kind_card",
            "memory_extraction_scope_card",
            "memory_extraction_confidence_card",
        ]
    elif family == "evidence":
        assert result["card"]["parsed"]["cand_001"]["support_kind"] == "direct"
        assert result["source_quote"] == SOURCE
        assert set(purposes) == {
            "memory_extraction_evidence_support_card",
            "memory_extraction_preserve_verbatim_card",
            "memory_extraction_candidate_language_card",
            "memory_extraction_source_reference_card",
        }
    elif family == "temporal":
        assert result["card"]["parsed"]["cand_001"]["type"] == "bounded"
        assert result["card"]["parsed"]["cand_001"]["valid_to_iso"]
        assert purposes == [
            "memory_extraction_temporal_type_card",
            "memory_extraction_temporal_interval_card",
        ]
    elif family == "language":
        assert result["cards"]["answer_language"]["parsed"]["answer_language"] == "en"
        assert result["prior_language_profile_applied"] is False
        assert purposes == [
            "need_detection_query_language_card",
            "need_detection_answer_language_card",
        ]
        assert "Known query language: en" in provider.requests[-1].messages[-1].content
    elif family == "beliefs":
        assert (
            result["key_generation"]["parsed"]["cand_001"]["claim_key"]
            == "projects.launch.deadline"
        )
        assert (
            result["frozen_key_comparator"]["candidate_key"]
            == "projects.launch.priority"
        )
        assert result["frozen_key_comparator"]["catalog_equivalence"] == {
            "projects.launch.ownership": False
        }
        comparator_request = provider.requests[-1]
        comparator_payload = "\n".join(
            message.content for message in comparator_request.messages
        )
        assert "projects.launch.priority" in comparator_payload
        assert "projects.launch.deadline" not in comparator_payload
        assert purposes == [
            "memory_extraction_belief_key_card",
            "memory_extraction_belief_value_card",
            "intent_classifier_claim_key_equivalence_batch",
        ]
    elif family == "members":
        assert len(result["card"]["parsed"]["cand_001"]) == 2
        assert result["identity_basis"] == "generated_member_list"
        assert purposes.count("memory_extraction_coverage_members_card") == 1
        assert purposes.count("memory_extraction_coverage_member_identity_card") == 2
    else:
        assert result["plan"]["actions"][0]["topic_id"] == "tpc_launch"
        assert len(purposes) == 7
        assert "Launch checklist" in provider.requests[0].messages[-1].content


@pytest.mark.asyncio
async def test_card_replay_rejects_real_cases_before_dispatch() -> None:
    provider = ReplayProvider()
    client = LLMClient(providers=[provider], structured_output_retry_attempts=0)
    case = _case("classification")
    case["origin"] = "aurvek_local_snapshot"
    with pytest.raises(ValueError, match="synthetic"):
        await run_card_replay(
            client, case=case, fixture=_fixture(), settings=_settings()
        )
    assert provider.requests == []


@pytest.mark.asyncio
async def test_card_replay_preserves_nonempty_recent_context() -> None:
    provider = ReplayProvider()
    client = LLMClient(providers=[provider], structured_output_retry_attempts=0)
    case = _case("classification")
    case["context_messages"] = [
        {
            "role": "assistant",
            "content": "The team discussed a previous checklist.",
        }
    ]
    await run_card_replay(client, case=case, fixture=_fixture(), settings=_settings())
    for request in provider.requests:
        assert "The team discussed a previous checklist." in "\n".join(
            message.content for message in request.messages
        )


@pytest.mark.asyncio
async def test_language_replay_supplies_frozen_prior_response_language() -> None:
    class ProfileProvider(ReplayProvider):
        async def complete(
            self, request: LLMCompletionRequest
        ) -> LLMCompletionResponse:
            self.requests.append(request)
            answer = (
                "unknown"
                if request.metadata["purpose"] == "need_detection_query_language_card"
                else "fr"
            )
            return LLMCompletionResponse(
                provider=self.name, model=request.model, output_text=answer
            )

    provider = ProfileProvider()
    case = _case("language")
    case["source_text"] = "42?"
    fixture = _fixture()
    fixture["card_replay_candidate"] = "42?"
    fixture["prior_language_profile"] = {"response_language": "fr"}
    result = await run_card_replay(
        LLMClient(providers=[provider], structured_output_retry_attempts=0),
        case=case,
        fixture=fixture,
        settings=_settings(),
    )
    assert result["prior_language_profile_applied"] is True
    assert result["cards"]["query_language"]["parsed"] == {"query_language": None}
    assert result["cards"]["answer_language"]["parsed"] == {"answer_language": "fr"}
    answer_prompt = provider.requests[-1].messages[-1].content
    assert "fr/default_answer_language/default" in answer_prompt
    assert "Known query language: unknown" in answer_prompt


@pytest.mark.asyncio
async def test_card_replay_uses_typed_choice_and_score_when_selected() -> None:
    class TypedProvider(LLMProvider):
        name = "typesafe"
        supports_choices = True
        supports_scores = True

        def __init__(self) -> None:
            self.requests: list[LLMCompletionRequest] = []

        async def complete(
            self, request: LLMCompletionRequest
        ) -> LLMCompletionResponse:
            self.requests.append(request)
            if request.score_questions:
                return LLMCompletionResponse(
                    provider=self.name,
                    model=request.model,
                    score_answers={
                        key: ScoreAnswer(
                            type="score",
                            score=3.2,
                            legend={
                                str(index): level
                                for index, level in enumerate(question.criteria)
                            },
                            probabilities={
                                "0": 0.0,
                                "1": 0.0,
                                "2": 0.0,
                                "3": 0.8,
                                "4": 0.2,
                            },
                            confidence=0.1,
                        )
                        for key, question in request.score_questions.items()
                    },
                )
            choice = (
                "evidence"
                if request.metadata["purpose"] == "memory_extraction_kind_card"
                else "user"
            )
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                choice_answers={
                    key: ChoiceAnswer(
                        type="choice",
                        name=key,
                        choice=choice,
                        probabilities={choice: 1.0},
                        confidence=1.0,
                    )
                    for key in request.choice_questions or {}
                },
            )

        async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
            raise AssertionError("Card replay must not embed")

    provider = TypedProvider()
    settings = _settings()
    for component in ("extraction_kind", "extraction_scope", "extraction_confidence"):
        settings.llm_component_models[component] = "typesafe/jev-1.13.0"
    settings.llm_finite_decisions_enabled = True
    result = await run_card_replay(
        LLMClient(providers=[provider], structured_output_retry_attempts=0),
        case=_case("classification"),
        fixture=_fixture(),
        settings=settings,
    )
    assert result["cards"]["memory_confidence"]["parsed"] == {"cand_001": 0.8}
    assert [bool(request.score_questions) for request in provider.requests] == [
        False,
        False,
        True,
    ]


@pytest.mark.asyncio
async def test_empty_belief_catalog_only_runs_generation() -> None:
    provider = ReplayProvider()
    client = LLMClient(providers=[provider], structured_output_retry_attempts=0)
    fixture = _fixture()
    fixture["belief"]["catalog_keys"] = []
    result = await run_card_replay(
        client, case=_case("beliefs"), fixture=fixture, settings=_settings()
    )
    assert result["frozen_key_comparator"]["catalog_equivalence"] == {}
    assert result["key_generation"]["parsed"]["cand_001"]["claim_key"]
    assert [request.metadata["purpose"] for request in provider.requests] == [
        "memory_extraction_belief_key_card", "memory_extraction_belief_value_card"
    ]
