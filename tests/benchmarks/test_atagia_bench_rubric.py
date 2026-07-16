"""Tests for Atagia-bench rubric alignment with the memory_quality doctrine.

Covers: the persona transcript renderer, the measurement-layer classification,
the llm_judge grader threading the conversation transcript to the memory_quality
judge, and the runner wiring (default protocol, grader-config transcript, and the
measurement-layer question filter).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from atagia.models.schemas_replay import AblationConfig
from atagia.services.llm_client import LLMCompletionRequest, LLMCompletionResponse
from benchmarks.atagia_bench.adapter import (
    AtagiaBenchConversation,
    AtagiaBenchDataset,
    AtagiaBenchPersona,
    AtagiaBenchPersonaData,
    AtagiaBenchQuestion,
    render_persona_transcript,
)
from benchmarks.atagia_bench.graders import (
    MEASUREMENT_LAYER_MEMORY_CONTENT,
    MEASUREMENT_LAYER_PRODUCT_BEHAVIOR,
    LLMJudgeGrader,
    measurement_layer_for_grader,
)
from benchmarks.atagia_bench.runner import AtagiaBenchRunner
from benchmarks.scorer import JudgeProtocol, LLMJudgeScorer


class _FakeLLMClient:
    """Capture the judge request and return a scripted verdict."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        return LLMCompletionResponse(
            provider="kimi",
            model="kimi-k2.7-code-highspeed",
            output_text=json.dumps(self._payload),
        )


def _persona_data() -> AtagiaBenchPersonaData:
    persona = AtagiaBenchPersona(
        persona_id="persona_a",
        display_name="Persona A",
        age=40,
        occupation="tester",
        profile="A synthetic persona.",
        modes=["default"],
        conversation_count=1,
        test_scenarios=["recall"],
    )
    conversation = AtagiaBenchConversation(
        conversation_id="persona_a-conv-01",
        mode="default",
        timestamp_base="2025-01-01T00:00:00",
        turns=[
            {
                "turn_id": "persona_a-01-t01",
                "role": "user",
                "text": "My dog is named Pluto and I live in Lyon.",
                "timestamp": "2025-01-01T09:00:00",
            },
            {
                "turn_id": "persona_a-01-t02",
                "role": "assistant",
                "text": "Noted: Pluto, and you are in Lyon.",
                "timestamp": "2025-01-01T09:00:05",
            },
        ],
    )
    question = AtagiaBenchQuestion(
        question_id="persona_a-q01",
        question_text="What is the name of the user's dog?",
        ground_truth="Pluto",
        answer_type="llm_judge",
        category_tags=["recall"],
        evidence_turn_ids=["persona_a-01-t01"],
        grader="llm_judge",
    )
    return AtagiaBenchPersonaData(
        persona=persona,
        conversations=[conversation],
        questions=[question],
    )


def test_render_persona_transcript_includes_turns_and_roles() -> None:
    transcript = render_persona_transcript(_persona_data().conversations)
    assert "conversation persona_a-conv-01" in transcript
    assert "user: My dog is named Pluto and I live in Lyon." in transcript
    assert "assistant: Noted: Pluto, and you are in Lyon." in transcript
    # Turn ids and timestamps are rendered so the judge can anchor evidence.
    assert "persona_a-01-t01" in transcript


def test_measurement_layer_classification() -> None:
    assert measurement_layer_for_grader("llm_judge") == MEASUREMENT_LAYER_MEMORY_CONTENT
    assert measurement_layer_for_grader("supersession") == MEASUREMENT_LAYER_MEMORY_CONTENT
    assert measurement_layer_for_grader("exact_match") == MEASUREMENT_LAYER_MEMORY_CONTENT
    assert measurement_layer_for_grader("set") == MEASUREMENT_LAYER_MEMORY_CONTENT
    assert measurement_layer_for_grader("normalized_date") == MEASUREMENT_LAYER_MEMORY_CONTENT
    assert measurement_layer_for_grader("abstention") == MEASUREMENT_LAYER_PRODUCT_BEHAVIOR
    assert measurement_layer_for_grader("gated") == MEASUREMENT_LAYER_PRODUCT_BEHAVIOR


def test_measurement_layer_unknown_grader_fails_fast() -> None:
    with pytest.raises(ValueError, match="Unclassified grader"):
        measurement_layer_for_grader("mystery")


@pytest.mark.asyncio
async def test_llm_judge_grader_passes_transcript_to_memory_quality() -> None:
    client = _FakeLLMClient(
        {
            "verdict": 1,
            "missing_info": False,
            "false_addition": False,
            "misattribution": False,
            "true_addition_only": True,
            "failure_reason": None,
            "reasoning": "Recalls Pluto; the Lyon extra is true in the conversation.",
        }
    )
    scorer = LLMJudgeScorer(
        client, "kimi/kimi-k2.7-code-highspeed", JudgeProtocol.MEMORY_QUALITY
    )
    grader = LLMJudgeGrader(scorer)
    transcript = render_persona_transcript(_persona_data().conversations)
    grade = await grader.grade(
        prediction="Pluto. You also mentioned you live in Lyon.",
        ground_truth="Pluto",
        config={
            "question_text": "What is the name of the user's dog?",
            "source_evidence": [
                {"turn_id": "persona_a-01-t01", "text": "My dog is named Pluto."}
            ],
            "conversation_transcript": transcript,
        },
    )
    assert grade.passed is True
    user_content = client.requests[0].messages[1].content
    assert user_content.startswith("Full source conversation:")
    assert "Pluto and I live in Lyon" in user_content


@pytest.mark.asyncio
async def test_llm_judge_memory_quality_without_transcript_fails_fast() -> None:
    client = _FakeLLMClient({"verdict": 1})
    scorer = LLMJudgeScorer(
        client, "kimi/kimi-k2.7-code-highspeed", JudgeProtocol.MEMORY_QUALITY
    )
    grader = LLMJudgeGrader(scorer)
    with pytest.raises(ValueError, match="conversation_transcript"):
        await grader.grade(
            prediction="Pluto",
            ground_truth="Pluto",
            config={"question_text": "What is the dog's name?"},
        )


def test_grader_config_adds_transcript_for_llm_judge_only() -> None:
    persona_data = _persona_data()
    llm_question = persona_data.questions[0]
    ablation = AblationConfig(privacy_enforcement="off")

    llm_config = AtagiaBenchRunner._grader_config_for_question(
        llm_question,
        ablation,
        persona_data=persona_data,
        answer_stance="reactive",
    )
    assert "conversation_transcript" in llm_config
    assert "user: My dog is named Pluto" in llm_config["conversation_transcript"]

    date_question = llm_question.model_copy(
        update={"grader": "normalized_date", "answer_type": "normalized_date"}
    )
    date_config = AtagiaBenchRunner._grader_config_for_question(
        date_question,
        ablation,
        persona_data=persona_data,
        answer_stance="reactive",
    )
    assert "conversation_transcript" not in date_config


def test_runner_default_judge_protocol_is_memory_quality() -> None:
    runner = AtagiaBenchRunner(
        llm_provider="openai",
        llm_api_key="test-key",
        llm_model="answer-model",
        judge_model="judge-model",
    )
    assert runner._judge_protocol is JudgeProtocol.MEMORY_QUALITY


def _dataset_with_layers() -> AtagiaBenchDataset:
    persona_data = _persona_data()
    extra = persona_data.questions[0].model_copy(
        update={
            "question_id": "persona_a-q02",
            "grader": "abstention",
            "answer_type": "abstention",
        }
    )
    persona_data = persona_data.model_copy(
        update={"questions": [persona_data.questions[0], extra]}
    )
    return AtagiaBenchDataset(personas=[persona_data])


def test_measurement_layer_filter_selects_memory_content() -> None:
    dataset = _dataset_with_layers()
    memory_ids = AtagiaBenchRunner._question_ids_for_measurement_layers(
        dataset, ["memory_content"]
    )
    assert memory_ids == {"persona_a-q01"}

    product_ids = AtagiaBenchRunner._question_ids_for_measurement_layers(
        dataset, ["product_behavior"]
    )
    assert product_ids == {"persona_a-q02"}

    assert AtagiaBenchRunner._question_ids_for_measurement_layers(dataset, None) is None


def test_measurement_layer_filter_rejects_unknown_layer() -> None:
    dataset = _dataset_with_layers()
    with pytest.raises(ValueError, match="Unknown measurement layer"):
        AtagiaBenchRunner._question_ids_for_measurement_layers(dataset, ["bogus"])


def test_third_party_runner_uses_same_rubric() -> None:
    """Competitor runs must default to the same memory_quality rubric."""
    third_party_runner = pytest.importorskip(
        "benchmarks.third_party.runner",
        reason="third-party benchmark runner is not present in this checkout",
    )
    ThirdPartyBenchRunner = third_party_runner.ThirdPartyBenchRunner

    runner = ThirdPartyBenchRunner(
        system_factory=lambda config: None,  # type: ignore[arg-type, return-value]
        llm_provider="openai",
        llm_api_key="test-key",
        llm_model="answer-model",
        judge_model="judge-model",
        llm_client=object(),  # type: ignore[arg-type]
    )
    assert runner._judge_protocol is JudgeProtocol.MEMORY_QUALITY

    persona_data = _persona_data()
    config = runner._grader_config_for_question(
        persona_data.questions[0],
        persona_data,
    )
    assert "conversation_transcript" in config
    assert "user: My dog is named Pluto" in config["conversation_transcript"]
