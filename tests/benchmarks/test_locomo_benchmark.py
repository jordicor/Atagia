"""Tests for the LoCoMo benchmark runner."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from benchmarks.base import BenchmarkQuestion, BenchmarkReport, ConversationReport, QuestionResult, ScoreResult
from benchmarks.locomo.__main__ import _format_report_summary
from benchmarks.locomo.benchmark import LoCoMoBenchmark
from atagia import Atagia
from atagia.models.schemas_replay import AblationConfig
from atagia.services.retrieval_service import RetrievalService
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)

MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"
_MEMORY_ID_PATTERN = re.compile(r'memory_id="([^"]+)"')


class BenchmarkProvider(LLMProvider):
    name = "locomo-benchmark-tests"

    def __init__(self) -> None:
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        purpose = str(request.metadata.get("purpose"))
        if purpose == "need_detection":
            return self._response(request, json.dumps([]))
        if purpose == "applicability_scoring":
            memory_ids = _MEMORY_ID_PATTERN.findall(request.messages[1].content)
            payload = [
                {"memory_id": memory_id, "llm_applicability": 0.9}
                for memory_id in memory_ids
            ]
            return self._response(request, json.dumps(payload))
        if purpose == "memory_extraction":
            message_text = request.messages[-1].content
            canonical_text = self._extract_canonical_text(message_text)
            if canonical_text is None:
                payload = {
                    "evidences": [],
                    "beliefs": [],
                    "contract_signals": [],
                    "state_updates": [],
                    "mode_guess": None,
                    "nothing_durable": True,
                }
            else:
                payload = {
                    "evidences": [
                        {
                            "canonical_text": canonical_text,
                            "scope": "conversation",
                            "confidence": 0.95,
                            "source_kind": "extracted",
                            "privacy_level": 0,
                            "payload": {"kind": "fact"},
                        }
                    ],
                    "beliefs": [],
                    "contract_signals": [],
                    "state_updates": [],
                    "mode_guess": None,
                    "nothing_durable": False,
                }
            return self._response(request, json.dumps(payload))
        if purpose == "contract_projection":
            return self._response(
                request,
                json.dumps({"signals": [], "nothing_durable": True}),
            )
        if purpose == "consequence_detection":
            return self._response(
                request,
                json.dumps(
                    {
                        "is_consequence": False,
                        "action_description": "",
                        "outcome_description": "",
                        "outcome_sentiment": "neutral",
                        "confidence": 0.0,
                        "likely_action_message_id": None,
                    }
                ),
            )
        if purpose == "benchmark_answer_generation":
            question = str(request.metadata.get("question"))
            answer_map = {
                "What color notebooks does Alice keep?": "red notebooks",
                "Where is Bob's mug stored?": "kitchen",
                "What color is the studio lamp?": "yellow",
            }
            return self._response(request, answer_map.get(question, "unknown"))
        if purpose == "benchmark_judge":
            prompt = request.messages[1].content
            lines = {
                line.split(": ", 1)[0]: line.split(": ", 1)[1]
                for line in prompt.splitlines()
                if ": " in line
            }
            prediction = lines.get("Prediction", "").strip().lower()
            ground_truth = lines.get("Ground truth", "").strip().lower()
            verdict = 1 if prediction == ground_truth else 0
            reasoning = "Matches ground truth." if verdict == 1 else "Does not match ground truth."
            return self._response(
                request,
                json.dumps({"verdict": verdict, "reasoning": reasoning}),
            )
        if purpose == "intent_classifier_explicit":
            return self._response(
                request,
                json.dumps({"is_explicit": True, "reasoning": "Benchmark stub response."}),
            )
        if purpose == "intent_classifier_claim_key_equivalence":
            return self._response(request, json.dumps({"equivalent": True}))
        raise AssertionError(f"Unexpected benchmark LLM purpose: {purpose}")

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embeddings are not used in benchmark tests")

    @staticmethod
    def _response(request: LLMCompletionRequest, output_text: str) -> LLMCompletionResponse:
        return LLMCompletionResponse(
            provider=BenchmarkProvider.name,
            model=request.model,
            output_text=output_text,
        )

    @staticmethod
    def _extract_canonical_text(prompt_text: str) -> str | None:
        for candidate in (
            "Alice keeps red notebooks in the studio.",
            "Bob stores the blue mug in the kitchen.",
            "The studio lamp is green.",
        ):
            if candidate in prompt_text:
                return candidate
        return None


def _install_stub_client(monkeypatch: pytest.MonkeyPatch, provider: BenchmarkProvider) -> None:
    monkeypatch.setattr(
        "atagia.app.build_llm_client",
        lambda _settings: LLMClient(provider_name=provider.name, providers=[provider]),
    )


def _write_dataset(tmp_path: Path) -> Path:
    data_path = tmp_path / "locomo-mini.json"
    data_path.write_text(
        json.dumps(
            [
                {
                    "sample_id": "conv-test-1",
                    "conversation": {
                        "speaker_a": "Alice",
                        "speaker_b": "Bob",
                        "session_1": [
                            {
                                "speaker": "Alice",
                                "dia_id": "D1:1",
                                "text": "Alice keeps red notebooks in the studio.",
                            },
                            {
                                "speaker": "Bob",
                                "dia_id": "D1:2",
                                "text": "I will remember your notebooks.",
                            },
                        ],
                        "session_1_date_time": "1:56 pm on 8 May, 2023",
                        "session_2": [
                            {
                                "speaker": "Alice",
                                "dia_id": "D2:1",
                                "text": "Bob stores the blue mug in the kitchen.",
                            },
                            {
                                "speaker": "Bob",
                                "dia_id": "D2:2",
                                "text": "Thanks for the reminder about the mug.",
                            },
                        ],
                        "session_2_date_time": "2:10 pm on 9 May, 2023",
                        "session_3": [
                            {
                                "speaker": "Alice",
                                "dia_id": "D3:1",
                                "text": "The studio lamp is green.",
                            },
                            {
                                "speaker": "Bob",
                                "dia_id": "D3:2",
                                "text": "That lamp color sounds nice.",
                            },
                        ],
                        "session_3_date_time": "3:10 pm on 10 May, 2023",
                    },
                    "qa": [
                        {
                            "question": "What color notebooks does Alice keep?",
                            "answer": "red notebooks",
                            "evidence": ["D1:1"],
                            "category": 1,
                        },
                        {
                            "question": "Where is Bob's mug stored?",
                            "answer": "kitchen",
                            "evidence": ["D2:1"],
                            "category": 2,
                        },
                        {
                            "question": "What color is the studio lamp?",
                            "answer": "green",
                            "evidence": ["D3:1"],
                            "category": 3,
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    return data_path


@pytest.mark.asyncio
async def test_benchmark_single_conversation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = BenchmarkProvider()
    _install_stub_client(monkeypatch, provider)
    benchmark = LoCoMoBenchmark(
        data_path=_write_dataset(tmp_path),
        llm_provider="openai",
        llm_api_key="test-openai-key",
        llm_model="answer-model",
        judge_model="judge-model",
        manifests_dir=MANIFESTS_DIR,
    )

    report = await benchmark.run()

    assert report.benchmark_name == "LoCoMo"
    assert report.total_questions == 3
    assert report.total_correct == 2
    assert report.overall_accuracy == pytest.approx(2 / 3)
    assert report.category_breakdown == {1: 1.0, 2: 1.0, 3: 0.0}
    assert len(report.conversations) == 1
    conversation_report = report.conversations[0]
    assert conversation_report.conversation_id == "conv-test-1"
    assert conversation_report.accuracy == pytest.approx(2 / 3)
    assert len(conversation_report.results) == 3
    assert report.model_info["provider"] == "openai"
    assert report.model_info["answer_model"] == "answer-model"
    assert report.model_info["judge_model"] == "judge-model"
    assert sum(
        1
        for request in provider.requests
        if request.metadata.get("purpose") == "need_detection"
    ) == 3
    assert sum(
        1
        for request in provider.requests
        if request.metadata.get("purpose") == "memory_extraction"
    ) == 6


@pytest.mark.asyncio
async def test_benchmark_with_ablation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = BenchmarkProvider()
    observed_ablation: list[AblationConfig | None] = []
    _install_stub_client(monkeypatch, provider)
    original_retrieve = RetrievalService.retrieve

    async def _capture_retrieve(
        self: RetrievalService,
        user_id: str,
        conversation_id: str,
        message_text: str,
        mode: str | None = None,
        ablation: AblationConfig | None = None,
    ):
        if message_text == "What color notebooks does Alice keep?":
            observed_ablation.append(ablation)
        return await original_retrieve(
            self,
            user_id=user_id,
            conversation_id=conversation_id,
            message_text=message_text,
            mode=mode,
            ablation=ablation,
        )

    monkeypatch.setattr(RetrievalService, "retrieve", _capture_retrieve)
    benchmark = LoCoMoBenchmark(
        data_path=_write_dataset(tmp_path),
        llm_provider="openai",
        llm_api_key="test-openai-key",
        llm_model="answer-model",
        judge_model="judge-model",
        manifests_dir=MANIFESTS_DIR,
    )

    report = await benchmark.run(
        ablation=AblationConfig(skip_applicability_scoring=True),
        conversation_ids=["conv-test-1"],
        categories=[1],
    )

    assert report.total_questions == 1
    assert report.ablation_config is not None
    assert report.ablation_config["skip_applicability_scoring"] is True
    assert len(observed_ablation) == 1
    assert observed_ablation[0] is not None
    assert observed_ablation[0].skip_applicability_scoring is True
    assert any(
        request.metadata.get("purpose") == "benchmark_answer_generation"
        for request in provider.requests
    )


@pytest.mark.asyncio
async def test_benchmark_max_questions_limits_validation_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = BenchmarkProvider()
    _install_stub_client(monkeypatch, provider)
    benchmark = LoCoMoBenchmark(
        data_path=_write_dataset(tmp_path),
        llm_provider="openai",
        llm_api_key="test-openai-key",
        llm_model="answer-model",
        judge_model="judge-model",
        manifests_dir=MANIFESTS_DIR,
    )

    report = await benchmark.run(max_questions=1)

    assert report.total_questions == 1
    assert len(report.conversations[0].results) == 1
    assert sum(
        1
        for request in provider.requests
        if request.metadata.get("purpose") == "need_detection"
    ) == 1
    assert sum(
        1
        for request in provider.requests
        if request.metadata.get("purpose") == "memory_extraction"
    ) == 6


@pytest.mark.asyncio
async def test_benchmark_max_turns_limits_ingestion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = BenchmarkProvider()
    ingested_messages: list[tuple[str, str, str | None]] = []
    _install_stub_client(monkeypatch, provider)
    original_ingest_message = Atagia.ingest_message

    async def _capture_ingest_message(
        self: Atagia,
        user_id: str,
        conversation_id: str,
        role: str,
        text: str,
        mode: str | None = None,
        workspace_id: str | None = None,
        occurred_at: str | None = None,
    ) -> None:
        ingested_messages.append((role, text, occurred_at))
        await original_ingest_message(
            self,
            user_id=user_id,
            conversation_id=conversation_id,
            role=role,
            text=text,
            mode=mode,
            workspace_id=workspace_id,
            occurred_at=occurred_at,
        )

    monkeypatch.setattr(Atagia, "ingest_message", _capture_ingest_message)
    benchmark = LoCoMoBenchmark(
        data_path=_write_dataset(tmp_path),
        llm_provider="openai",
        llm_api_key="test-openai-key",
        llm_model="answer-model",
        judge_model="judge-model",
        manifests_dir=MANIFESTS_DIR,
    )

    report = await benchmark.run(max_turns=2)

    assert report.total_questions == 3
    assert ingested_messages == [
        ("user", "Alice: Alice keeps red notebooks in the studio.", "2023-05-08T13:56:00"),
        ("assistant", "Bob: I will remember your notebooks.", "2023-05-08T13:56:00"),
    ]


def test_format_report_summary_includes_key_sections(tmp_path: Path) -> None:
    report = BenchmarkReport(
        benchmark_name="LoCoMo",
        overall_accuracy=0.5,
        category_breakdown={1: 0.5},
        conversations=[
            ConversationReport(
                conversation_id="conv-test-1",
                results=[
                    QuestionResult(
                        question=BenchmarkQuestion(
                            question_text="Question?",
                            ground_truth="answer",
                            category=1,
                            evidence_turn_ids=["D1:1"],
                            question_id="conv-test-1:q1",
                        ),
                        prediction="answer",
                        score_result=ScoreResult(
                            score=1,
                            reasoning="ok",
                            judge_model="judge-model",
                        ),
                        memories_used=1,
                        retrieval_time_ms=12.0,
                    ),
                    QuestionResult(
                        question=BenchmarkQuestion(
                            question_text="Question 2?",
                            ground_truth="answer 2",
                            category=1,
                            evidence_turn_ids=["D1:2"],
                            question_id="conv-test-1:q2",
                        ),
                        prediction="wrong",
                        score_result=ScoreResult(
                            score=0,
                            reasoning="no",
                            judge_model="judge-model",
                        ),
                        memories_used=0,
                        retrieval_time_ms=11.0,
                    ),
                ],
                accuracy=0.5,
                category_breakdown={1: 0.5},
            )
        ],
        total_questions=2,
        total_correct=1,
        ablation_config=None,
        timestamp="2026-04-01T00:00:00+00:00",
        model_info={
            "provider": "openai",
            "answer_model": "answer-model",
            "judge_model": "judge-model",
        },
        duration_seconds=83.0,
    )
    summary = _format_report_summary(
        report=report,
        report_path=tmp_path / "report.json",
    )

    assert "LoCoMo Benchmark Results" in summary
    assert "Overall accuracy: 50.0% (1/2)" in summary
    assert "Duration: 1m 23s" in summary
    assert "Model: openai / answer-model" in summary
    assert "Cat 1 (single-hop):" in summary
    assert "Per-conversation:" in summary
    assert "conv-test-1:" in summary
    assert "Report saved to:" in summary
