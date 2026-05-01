"""Tests for the LoCoMo benchmark runner."""

from __future__ import annotations

import json
import hashlib
import re
import sqlite3
from pathlib import Path

import pytest

from benchmarks.base import BenchmarkQuestion, BenchmarkReport, ConversationReport, QuestionResult, ScoreResult
from benchmarks.locomo.__main__ import (
    _build_parser,
    _format_benchmark_db_list,
    _format_benchmark_db_list_json,
    _format_benchmark_db_snapshot_diff,
    _format_benchmark_db_snapshot_diff_json,
    _format_report_summary,
    _format_run_log_summary,
    _format_run_log_summary_json,
)
from benchmarks.locomo.benchmark import LoCoMoBenchmark
from atagia import Atagia
from atagia.core.repositories import MessageRepository
from atagia.models.schemas_replay import AblationConfig
from atagia.services.retrieval_service import RetrievalService
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMError,
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
            return self._response(
                request,
                json.dumps(
                    {
                        "needs": [],
                        "temporal_range": None,
                        "sub_queries": ["locomo benchmark"],
                        "sparse_query_hints": [
                            {
                                "sub_query_text": "locomo benchmark",
                                "fts_phrase": "locomo benchmark",
                            }
                        ],
                        "query_type": "default",
                        "retrieval_levels": [0],
                    }
                ),
            )
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
                            "language_codes": ["en"],
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
        if purpose == "summary_chunk_segmentation":
            prompt = request.messages[1].content
            message_sequences = [int(item) for item in re.findall(r'<message seq="(\d+)"', prompt)]
            start_seq = min(message_sequences) if message_sequences else 1
            end_seq = max(message_sequences) if message_sequences else 1
            return self._response(
                request,
                json.dumps(
                    {
                        "episodes": [
                            {
                                "start_seq": start_seq,
                                "end_seq": end_seq,
                                "summary_text": "Benchmark chunk summary.",
                            }
                        ]
                    }
                ),
            )
        if purpose == "episode_synthesis":
            prompt = request.messages[1].content
            chunk_ids = re.findall(r'<conversation_chunk id="([^"]+)"', prompt)
            return self._response(
                request,
                json.dumps(
                    {
                        "episodes": [
                            {
                                "episode_key": "benchmark",
                                "summary_text": "Benchmark episode summary.",
                            }
                        ],
                        "chunk_episode_keys": ["benchmark"] * len(chunk_ids),
                    }
                ),
            )
        if purpose == "thematic_profile_synthesis":
            return self._response(
                request,
                json.dumps(
                    {
                        "summary_text": "Benchmark thematic profile.",
                        "cited_episode_keys": ["benchmark"],
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
        prompt_tokens = sum(len(message.content.split()) for message in request.messages)
        completion_tokens = max(1, len(output_text.split()))
        return LLMCompletionResponse(
            provider=BenchmarkProvider.name,
            model=request.model,
            output_text=output_text,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
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


class FailingSecondAnswerProvider(BenchmarkProvider):
    """Provider that fails after the first scored answer is checkpointed."""

    def __init__(self) -> None:
        super().__init__()
        self.answer_generation_calls = 0

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        if request.metadata.get("purpose") == "benchmark_answer_generation":
            self.answer_generation_calls += 1
            if self.answer_generation_calls == 2:
                raise RuntimeError("simulated benchmark interruption")
        return await super().complete(request)


class JudgeFailingForOneQuestionProvider(BenchmarkProvider):
    """Provider whose judge call raises LLMError for one targeted question."""

    def __init__(self, failing_question_text: str) -> None:
        super().__init__()
        self._failing_question_text = failing_question_text
        self.judge_failures = 0

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        if (
            request.metadata.get("purpose") == "benchmark_judge"
            and request.metadata.get("question") == self._failing_question_text
        ):
            self.judge_failures += 1
            raise LLMError(
                "Anthropic stopped because it reached max output tokens"
            )
        return await super().complete(request)


class AnswerFailingForOneQuestionProvider(BenchmarkProvider):
    """Provider whose benchmark answer call raises LLMError for one question."""

    def __init__(self, failing_question_text: str) -> None:
        super().__init__()
        self._failing_question_text = failing_question_text
        self.answer_failures = 0

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        if (
            request.metadata.get("purpose") == "benchmark_answer_generation"
            and request.metadata.get("question") == self._failing_question_text
        ):
            self.answer_failures += 1
            raise LLMError("Answer provider overloaded")
        return await super().complete(request)


def _install_stub_client(monkeypatch: pytest.MonkeyPatch, provider: BenchmarkProvider) -> None:
    monkeypatch.setattr(
        "atagia.app.build_llm_client",
        lambda _settings: LLMClient(provider_name=provider.name, providers=[provider]),
    )
    # LoCoMo asserts that need_detection runs for every question. Keep the
    # full retrieval pipeline active by disabling the small-corpus shortcut.
    monkeypatch.setenv("ATAGIA_SMALL_CORPUS_TOKEN_THRESHOLD_RATIO", "0")


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


def _write_dataset_with_image_caption(tmp_path: Path) -> Path:
    data_path = tmp_path / "locomo-image.json"
    data_path.write_text(
        json.dumps(
            [
                {
                    "sample_id": "conv-image",
                    "conversation": {
                        "speaker_a": "Alice",
                        "speaker_b": "Bob",
                        "session_1": [
                            {
                                "speaker": "Bob",
                                "dia_id": "D1:1",
                                "text": "The kids loved making something with clay. They made this!",
                                "img_url": ["https://example.test/dog-cup.jpg"],
                                "blip_caption": "a photo of a cup with a dog face on it",
                                "query": "kids pottery finished pieces",
                            },
                        ],
                        "session_1_date_time": "1:56 pm on 8 May, 2023",
                    },
                    "qa": [
                        {
                            "question": "What kind of pot did Bob and the kids make with clay?",
                            "answer": "a cup with a dog face on it",
                            "evidence": ["D1:1"],
                            "category": 4,
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    return data_path


def _write_two_conversation_dataset(tmp_path: Path) -> Path:
    data_path = _write_dataset(tmp_path)
    samples = json.loads(data_path.read_text(encoding="utf-8"))
    second_sample = json.loads(json.dumps(samples[0]))
    second_sample["sample_id"] = "conv-test-2"
    samples.append(second_sample)
    data_path.write_text(json.dumps(samples), encoding="utf-8")
    return data_path


def _write_dataset_with_missing_timestamp(tmp_path: Path) -> Path:
    data_path = _write_dataset(tmp_path)
    samples = json.loads(data_path.read_text(encoding="utf-8"))
    samples[0]["conversation"].pop("session_1_date_time")
    data_path.write_text(json.dumps(samples), encoding="utf-8")
    return data_path


def test_atagia_constructor_accepts_phase_model_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ATAGIA_LLM_FORCED_GLOBAL_MODEL", "")
    monkeypatch.setenv("ATAGIA_LLM_INGEST_MODEL", "")
    monkeypatch.setenv("ATAGIA_LLM_RETRIEVAL_MODEL", "")
    monkeypatch.setenv("ATAGIA_LLM_CHAT_MODEL", "")
    engine = Atagia(
        llm_ingest_model="openai/ingest-model,minimal",
        llm_retrieval_model="openai/retrieval-model,medium",
        llm_chat_model="openai/chat-model,high",
        llm_component_models={"extractor": "openai/extractor-model,none"},
    )

    settings = engine._build_settings()

    assert settings.llm_forced_global_model is None
    assert settings.llm_ingest_model == "openai/ingest-model,minimal"
    assert settings.llm_retrieval_model == "openai/retrieval-model,medium"
    assert settings.llm_chat_model == "openai/chat-model,high"
    assert settings.llm_component_models["extractor"] == "openai/extractor-model,none"


def test_locomo_cli_accepts_answer_model_and_component_overrides() -> None:
    args = _build_parser().parse_args(
        [
            "--data-path",
            "benchmarks/data/locomo10.json",
            "--provider",
            "openrouter",
            "--answer-model",
            "openrouter/google/gemini-3.1-flash-lite-preview,medium",
            "--ingest-model",
            "openrouter/google/gemini-3.1-flash-lite-preview,none",
            "--retrieval-model",
            "openrouter/google/gemini-3.1-flash-lite-preview,minimal",
            "--component-model",
            "extractor=openai/gpt-4o-mini",
            "--answer-prompt-variant",
            "grounded_connect",
        ]
    )

    assert args.model is None
    assert args.answer_model == "openrouter/google/gemini-3.1-flash-lite-preview,medium"
    assert args.ingest_model == "openrouter/google/gemini-3.1-flash-lite-preview,none"
    assert args.retrieval_model == "openrouter/google/gemini-3.1-flash-lite-preview,minimal"
    assert args.component_model == ["extractor=openai/gpt-4o-mini"]
    assert args.answer_prompt_variant == "grounded_connect"


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
    assert conversation_report.results[0].trace["diagnosis_bucket"] == "passed"
    assert conversation_report.results[0].trace["retrieval_trace"]["query_text"] == (
        "What color notebooks does Alice keep?"
    )
    assert conversation_report.results[0].trace["shadow_sufficiency_diagnostics"] is not None
    assert (
        conversation_report.results[0].trace["retrieval_trace"]["retrieval_sufficiency"]
        == conversation_report.results[0].trace["shadow_sufficiency_diagnostics"]
    )
    assert "selected_memory_ids" in conversation_report.results[0].trace
    assert report.model_info["provider"] == "openai"
    assert report.model_info["answer_model"] == "openai/answer-model"
    assert report.model_info["judge_model"] == "openai/judge-model"
    assert report.model_info["ingest_mode"] == "online"
    assert report.model_info["parallel_questions"] == 1
    assert report.model_info["benchmark_db"]["ingest_mode"] == "online"
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
    assert report.model_info["llm_call_summary"]["total_calls"] > 0
    assert report.conversations[0].results[0].trace["llm_calls"]


@pytest.mark.asyncio
async def test_benchmark_routes_models_by_phase_and_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = BenchmarkProvider()
    _install_stub_client(monkeypatch, provider)
    benchmark = LoCoMoBenchmark(
        data_path=_write_dataset(tmp_path),
        llm_provider="openai",
        llm_api_key="test-openai-key",
        llm_model=None,
        answer_model="answer-model,high",
        judge_model="judge-model,medium",
        ingest_model="ingest-model,minimal",
        retrieval_model="retrieval-model,none",
        chat_model_override="chat-model",
        component_models={"extractor": "extractor-model,minimal"},
        answer_prompt_variant="grounded_connect",
        manifests_dir=MANIFESTS_DIR,
    )

    report = await benchmark.run(categories=[1], max_questions=1)

    models_by_purpose: dict[str, set[str]] = {}
    for request in provider.requests:
        purpose = str(request.metadata.get("purpose"))
        models_by_purpose.setdefault(purpose, set()).add(request.model)

    assert models_by_purpose["memory_extraction"] == {"openai/extractor-model,minimal"}
    assert models_by_purpose["contract_projection"] == {"openai/ingest-model,minimal"}
    assert models_by_purpose["need_detection"] == {"openai/retrieval-model,none"}
    assert models_by_purpose["applicability_scoring"] == {"openai/retrieval-model,none"}
    assert models_by_purpose["benchmark_answer_generation"] == {"openai/answer-model,high"}
    assert models_by_purpose["benchmark_judge"] == {"openai/judge-model,medium"}

    assert report.model_info["answer_model"] == "openai/answer-model,high"
    assert report.model_info["judge_model"] == "openai/judge-model,medium"
    assert report.model_info["forced_global_model"] is None
    assert report.model_info["ingest_model"] == "openai/ingest-model,minimal"
    assert report.model_info["retrieval_model"] == "openai/retrieval-model,none"
    assert report.model_info["chat_model"] == "openai/chat-model"
    assert report.model_info["component_models"] == {
        "extractor": "openai/extractor-model,minimal"
    }
    assert report.model_info["answer_prompt_variant"] == "grounded_connect"
    answer_request = next(
        request
        for request in provider.requests
        if request.metadata.get("purpose") == "benchmark_answer_generation"
    )
    assert "Benchmark answer variant" in answer_request.messages[0].content
    llm_calls = report.conversations[0].results[0].trace["llm_calls"]
    assert {call["purpose"] for call in llm_calls} >= {
        "need_detection",
        "applicability_scoring",
        "benchmark_answer_generation",
        "benchmark_judge",
    }
    assert report.model_info["llm_call_summary"]["by_purpose"]["memory_extraction"]["calls"] == 6


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
        trace: object | None = None,
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
            trace=trace,
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
    assert report.model_info["selection"]["max_questions"] == 1
    assert report.model_info["selection"]["planned_question_count"] == 1
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
async def test_benchmark_question_ids_filter_validation_run(
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

    report = await benchmark.run(question_ids=["conv-test-1:q2"])

    assert report.total_questions == 1
    assert report.conversations[0].results[0].question.question_id == "conv-test-1:q2"
    assert report.model_info["selection"]["question_filter"] == ["conv-test-1:q2"]
    assert report.model_info["selection"]["planned_question_count"] == 1


@pytest.mark.asyncio
async def test_benchmark_writes_partial_checkpoint_before_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FailingSecondAnswerProvider()
    _install_stub_client(monkeypatch, provider)
    checkpoint_path = tmp_path / "checkpoints" / "locomo-checkpoint.json"
    benchmark = LoCoMoBenchmark(
        data_path=_write_dataset(tmp_path),
        llm_provider="openai",
        llm_api_key="test-openai-key",
        llm_model="answer-model",
        judge_model="judge-model",
        manifests_dir=MANIFESTS_DIR,
    )

    with pytest.raises(RuntimeError, match="simulated benchmark interruption"):
        await benchmark.run(max_questions=2, checkpoint_path=checkpoint_path)

    checkpoint = BenchmarkReport.model_validate_json(checkpoint_path.read_text())
    assert checkpoint.total_questions == 1
    assert checkpoint.total_correct == 1
    assert checkpoint.overall_accuracy == pytest.approx(1.0)
    assert len(checkpoint.conversations) == 1
    assert checkpoint.conversations[0].conversation_id == "conv-test-1"
    assert checkpoint.conversations[0].results[0].question.question_text == (
        "What color notebooks does Alice keep?"
    )
    assert checkpoint.model_info["checkpoint"] == {
        "partial": True,
        "conversation_id": "conv-test-1",
        "conversation_index": 1,
        "conversation_count": 1,
        "completed_questions": 1,
        "selected_questions": 2,
        "trusted_evaluation": False,
    }


@pytest.mark.asyncio
async def test_score_question_labels_retrieval_error_without_cancelling_siblings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failing_question_text = "What color notebooks does Alice keep?"
    provider = BenchmarkProvider()
    _install_stub_client(monkeypatch, provider)
    original_retrieve = RetrievalService.retrieve

    async def _fail_one_retrieval(
        self: RetrievalService,
        user_id: str,
        conversation_id: str,
        message_text: str,
        mode: str | None = None,
        ablation: AblationConfig | None = None,
        trace: object | None = None,
    ):
        if message_text == failing_question_text:
            raise LLMError("Retrieval provider overloaded")
        return await original_retrieve(
            self,
            user_id=user_id,
            conversation_id=conversation_id,
            message_text=message_text,
            mode=mode,
            ablation=ablation,
            trace=trace,
        )

    monkeypatch.setattr(RetrievalService, "retrieve", _fail_one_retrieval)
    benchmark = LoCoMoBenchmark(
        data_path=_write_dataset(tmp_path),
        llm_provider="openai",
        llm_api_key="test-openai-key",
        llm_model="answer-model",
        judge_model="judge-model",
        manifests_dir=MANIFESTS_DIR,
    )

    report = await benchmark.run(parallel_questions=2)

    assert report.total_questions == 3
    failed = next(
        result
        for result in report.conversations[0].results
        if result.question.question_text == failing_question_text
    )
    assert failed.score_result.score == 0
    assert "Retrieval failed" in failed.score_result.reasoning
    assert failed.trace["failure_stage"] == "retrieval"
    assert failed.trace["diagnosis_bucket"] == "retrieval_failed"
    assert failed.trace["sufficiency_diagnostic"] == "retrieval_failed"
    assert failed.trace["retrieval_failure"]["exception_class"] == "LLMError"
    assert report.model_info["warning_counts"]["retrieval_failed"] == 1
    surviving_results = [
        result
        for result in report.conversations[0].results
        if result.question.question_text != failing_question_text
    ]
    assert len(surviving_results) == 2
    assert all("failure_stage" not in result.trace for result in surviving_results)


@pytest.mark.asyncio
async def test_score_question_labels_answer_generation_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failing_question_text = "What color notebooks does Alice keep?"
    provider = AnswerFailingForOneQuestionProvider(failing_question_text)
    _install_stub_client(monkeypatch, provider)
    benchmark = LoCoMoBenchmark(
        data_path=_write_dataset(tmp_path),
        llm_provider="openai",
        llm_api_key="test-openai-key",
        llm_model="answer-model",
        judge_model="judge-model",
        manifests_dir=MANIFESTS_DIR,
    )

    report = await benchmark.run(parallel_questions=2)

    assert provider.answer_failures == 1
    failed = next(
        result
        for result in report.conversations[0].results
        if result.question.question_text == failing_question_text
    )
    assert failed.score_result.score == 0
    assert "Answer generation failed" in failed.score_result.reasoning
    assert failed.trace["failure_stage"] == "answer_generation"
    assert failed.trace["diagnosis_bucket"] == "answer_generation_failed"
    assert failed.trace["sufficiency_diagnostic"] == "answer_generation_failed"
    assert failed.trace["answer_generation_failure"]["exception_class"] == "LLMError"
    assert "retrieval_trace" in failed.trace
    assert report.model_info["warning_counts"]["answer_generation_failed"] == 1


@pytest.mark.asyncio
async def test_score_question_tolerates_judge_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failing_question_text = "What color notebooks does Alice keep?"
    provider = JudgeFailingForOneQuestionProvider(failing_question_text)
    _install_stub_client(monkeypatch, provider)
    benchmark = LoCoMoBenchmark(
        data_path=_write_dataset(tmp_path),
        llm_provider="openai",
        llm_api_key="test-openai-key",
        llm_model="answer-model",
        judge_model="judge-model",
        manifests_dir=MANIFESTS_DIR,
    )

    report = await benchmark.run(parallel_questions=2)

    assert provider.judge_failures == 1
    assert report.total_questions == 3
    # Two questions still got real verdicts; one was tolerated as a 0 with a
    # captured error reason.
    assert len(report.conversations) == 1
    results = report.conversations[0].results
    assert len(results) == 3
    failing_results = [
        result
        for result in results
        if result.question.question_text == failing_question_text
    ]
    assert len(failing_results) == 1
    failed = failing_results[0]
    assert failed.score_result.score == 0
    assert "Judge call failed" in failed.score_result.reasoning
    assert "LLMError" in failed.score_result.reasoning
    assert failed.score_result.judge_model == "openai/judge-model"
    assert failed.trace["failure_stage"] == "judge"
    assert failed.trace["diagnosis_bucket"] == "judge_failed"
    assert failed.trace["sufficiency_diagnostic"] == "judge_failed"
    assert failed.trace.get("judge_failure", {}).get("exception_class") == "LLMError"
    assert report.model_info["warning_counts"]["judge_failed"] == 1
    surviving_results = [
        result
        for result in results
        if result.question.question_text != failing_question_text
    ]
    assert len(surviving_results) == 2
    # Surviving questions retain their normal trace shape (judge_failure absent).
    for result in surviving_results:
        assert "judge_failure" not in result.trace
        assert result.score_result.judge_model == "openai/judge-model"


@pytest.mark.asyncio
async def test_benchmark_parallel_conversations_use_isolated_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = BenchmarkProvider()
    _install_stub_client(monkeypatch, provider)
    checkpoint_path = tmp_path / "checkpoints" / "locomo-checkpoint.json"
    benchmark = LoCoMoBenchmark(
        data_path=_write_two_conversation_dataset(tmp_path),
        llm_provider="openai",
        llm_api_key="test-openai-key",
        llm_model="answer-model",
        judge_model="judge-model",
        manifests_dir=MANIFESTS_DIR,
    )

    report = await benchmark.run(
        max_questions=1,
        checkpoint_path=checkpoint_path,
        parallel_conversations=2,
    )

    assert report.total_questions == 2
    assert report.total_correct == 2
    assert [conversation.conversation_id for conversation in report.conversations] == [
        "conv-test-1",
        "conv-test-2",
    ]
    assert report.model_info["parallel_conversations"] == 2

    final_checkpoint = BenchmarkReport.model_validate_json(checkpoint_path.read_text())
    assert final_checkpoint.total_questions == 2
    assert [conversation.conversation_id for conversation in final_checkpoint.conversations] == [
        "conv-test-1",
        "conv-test-2",
    ]

    for conversation_id in ("conv-test-1", "conv-test-2"):
        partial_path = checkpoint_path.with_name(
            f"{checkpoint_path.stem}-{conversation_id}{checkpoint_path.suffix}"
        )
        partial = BenchmarkReport.model_validate_json(partial_path.read_text())
        assert partial.total_questions == 1
        assert [conversation.conversation_id for conversation in partial.conversations] == [
            conversation_id
        ]


@pytest.mark.asyncio
async def test_benchmark_parallel_questions_preserve_order_and_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = BenchmarkProvider()
    _install_stub_client(monkeypatch, provider)
    checkpoint_path = tmp_path / "checkpoints" / "locomo-checkpoint.json"
    benchmark = LoCoMoBenchmark(
        data_path=_write_dataset(tmp_path),
        llm_provider="openai",
        llm_api_key="test-openai-key",
        llm_model="answer-model",
        judge_model="judge-model",
        manifests_dir=MANIFESTS_DIR,
    )

    report = await benchmark.run(
        checkpoint_path=checkpoint_path,
        parallel_questions=2,
    )

    question_ids = [
        result.question.question_id
        for result in report.conversations[0].results
    ]
    assert question_ids == ["conv-test-1:q1", "conv-test-1:q2", "conv-test-1:q3"]
    assert report.model_info["parallel_questions"] == 2

    checkpoint = BenchmarkReport.model_validate_json(checkpoint_path.read_text())
    checkpoint_question_ids = [
        result.question.question_id
        for result in checkpoint.conversations[0].results
    ]
    assert checkpoint_question_ids == question_ids
    assert checkpoint.model_info["parallel_questions"] == 2
    assert checkpoint.total_questions == 3


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
        attachments: list[dict[str, object]] | None = None,
        *,
        operational_profile: str | None = None,
        operational_signals: dict[str, object] | None = None,
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
            attachments=attachments,
            operational_profile=operational_profile,
            operational_signals=operational_signals,
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


@pytest.mark.asyncio
async def test_benchmark_bulk_ingest_rebuilds_without_per_turn_flush(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = BenchmarkProvider()
    _install_stub_client(monkeypatch, provider)
    data_path = _write_dataset(tmp_path)
    db_dir = tmp_path / "benchmark-dbs"
    flush_calls = 0
    original_flush = Atagia.flush

    async def _fail_ingest_message(
        self: Atagia,
        user_id: str,
        conversation_id: str,
        role: str,
        text: str,
        mode: str | None = None,
        workspace_id: str | None = None,
        occurred_at: str | None = None,
        attachments: list[dict[str, object]] | None = None,
        *,
        operational_profile: str | None = None,
        operational_signals: dict[str, object] | None = None,
    ) -> None:
        raise AssertionError("bulk ingest should persist messages without live enqueue")

    async def _capture_flush(self: Atagia, timeout_seconds: float = 30.0) -> bool:
        nonlocal flush_calls
        flush_calls += 1
        return await original_flush(self, timeout_seconds=timeout_seconds)

    monkeypatch.setattr(Atagia, "ingest_message", _fail_ingest_message)
    monkeypatch.setattr(Atagia, "flush", _capture_flush)
    benchmark = LoCoMoBenchmark(
        data_path=data_path,
        llm_provider="openai",
        llm_api_key="test-openai-key",
        llm_model="answer-model",
        judge_model="judge-model",
        manifests_dir=MANIFESTS_DIR,
    )

    report = await benchmark.run(
        conversation_ids=["conv-test-1"],
        categories=[1],
        max_questions=1,
        benchmark_db_dir=db_dir,
        keep_db=True,
        ingest_mode="bulk",
    )

    assert report.total_questions == 1
    assert report.model_info["ingest_mode"] == "bulk"
    assert report.model_info["parallel_questions"] == 1
    assert report.model_info["benchmark_db"]["ingest_mode"] == "bulk"
    assert flush_calls == 1
    retained_dbs = list(db_dir.glob("*/benchmark.db"))
    assert len(retained_dbs) == 1
    retained_db = retained_dbs[0]
    metadata = json.loads((retained_db.parent / "run_metadata.json").read_text())
    assert metadata["ingest_mode"] == "bulk"
    assert metadata["status"] == "complete"
    assert metadata["rebuild_result"]["processed_messages"] == 6
    assert "parallel_questions" not in metadata

    eval_provider = BenchmarkProvider()
    _install_stub_client(monkeypatch, eval_provider)
    eval_benchmark = LoCoMoBenchmark(
        data_path=data_path,
        llm_provider="openai",
        llm_api_key="test-openai-key",
        llm_model="answer-model",
        judge_model="judge-model",
        manifests_dir=MANIFESTS_DIR,
    )
    eval_report = await eval_benchmark.run(
        conversation_ids=["conv-test-1"],
        categories=[1],
        max_questions=1,
        reuse_db=retained_db,
    )

    assert eval_report.model_info["ingest_mode"] == "bulk"
    assert eval_report.model_info["requested_ingest_mode"] == "online"
    assert eval_report.model_info["benchmark_db"]["ingest_mode"] == "bulk"
    assert eval_report.conversations[0].metadata["ingest_mode"] == "bulk"
    assert sum(
        1
        for request in eval_provider.requests
        if request.metadata.get("purpose") == "memory_extraction"
    ) == 0
    assert sum(
        1
        for request in provider.requests
        if request.metadata.get("purpose") == "memory_extraction"
    ) == 6
    assert any(
        request.metadata.get("purpose") == "summary_chunk_segmentation"
        for request in provider.requests
    )

    async with Atagia(
        db_path=retained_db,
        manifests_dir=MANIFESTS_DIR,
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/answer-model",
    ) as engine:
        runtime = engine.runtime
        assert runtime is not None
        connection = await runtime.open_connection()
        try:
            messages = await MessageRepository(
                connection,
                runtime.clock,
            ).list_messages_for_conversation("conv-test-1", "benchmark-user")
        finally:
            await connection.close()

    assert len(messages) == 6
    assert all("What color notebooks" not in str(message["text"]) for message in messages)


@pytest.mark.asyncio
async def test_benchmark_bulk_ingest_persists_locomo_image_captions_as_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = BenchmarkProvider()
    _install_stub_client(monkeypatch, provider)
    data_path = _write_dataset_with_image_caption(tmp_path)
    db_dir = tmp_path / "benchmark-dbs"
    benchmark = LoCoMoBenchmark(
        data_path=data_path,
        llm_provider="openai",
        llm_api_key="test-openai-key",
        llm_model="answer-model",
        judge_model="judge-model",
        manifests_dir=MANIFESTS_DIR,
    )

    await benchmark.run(
        conversation_ids=["conv-image"],
        ingest_only=True,
        max_turns=1,
        benchmark_db_dir=db_dir,
        keep_db=True,
        ingest_mode="bulk",
    )

    retained_db = next(db_dir.glob("*/benchmark.db"))
    with sqlite3.connect(retained_db) as connection:
        connection.row_factory = sqlite3.Row
        message = connection.execute(
            "SELECT text, metadata_json FROM messages ORDER BY seq LIMIT 1"
        ).fetchone()
        artifact = connection.execute(
            "SELECT artifact_type, source_kind, source_ref, metadata_json FROM artifacts"
        ).fetchone()
        chunks = connection.execute(
            "SELECT text, kind FROM artifact_chunks ORDER BY chunk_index"
        ).fetchall()

    assert message is not None
    assert "a photo of a cup with a dog face on it" in message["text"]
    message_metadata = json.loads(message["metadata_json"])
    assert message_metadata["attachment_count"] == 1
    assert artifact is not None
    assert artifact["artifact_type"] == "image"
    assert artifact["source_kind"] == "url"
    assert artifact["source_ref"] == "https://example.test/dog-cup.jpg"
    artifact_metadata = json.loads(artifact["metadata_json"])
    assert artifact_metadata["source"] == "locomo"
    assert artifact_metadata["caption_kind"] == "blip_caption"
    assert artifact_metadata["turn_id"] == "D1:1"
    assert any("a photo of a cup with a dog face on it" in row["text"] for row in chunks)
    assert any(row["kind"] == "parsed" for row in chunks)
    assert any(
        "a photo of a cup with a dog face on it" in request.messages[-1].content
        for request in provider.requests
        if request.metadata.get("purpose") == "memory_extraction"
    )


@pytest.mark.asyncio
async def test_benchmark_bulk_ingest_falls_back_for_missing_turn_timestamp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = BenchmarkProvider()
    _install_stub_client(monkeypatch, provider)
    db_dir = tmp_path / "benchmark-dbs"
    benchmark = LoCoMoBenchmark(
        data_path=_write_dataset_with_missing_timestamp(tmp_path),
        llm_provider="openai",
        llm_api_key="test-openai-key",
        llm_model="answer-model",
        judge_model="judge-model",
        manifests_dir=MANIFESTS_DIR,
    )

    await benchmark.run(
        conversation_ids=["conv-test-1"],
        categories=[1],
        max_questions=1,
        benchmark_db_dir=db_dir,
        keep_db=True,
        ingest_mode="bulk",
    )

    retained_db = next(db_dir.glob("*/benchmark.db"))
    async with Atagia(
        db_path=retained_db,
        manifests_dir=MANIFESTS_DIR,
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/answer-model",
    ) as engine:
        runtime = engine.runtime
        assert runtime is not None
        connection = await runtime.open_connection()
        try:
            messages = await MessageRepository(
                connection,
                runtime.clock,
            ).list_messages_for_conversation("conv-test-1", "benchmark-user")
        finally:
            await connection.close()

    assert messages[0]["occurred_at"] is not None
    assert str(messages[0]["occurred_at"]).strip()


@pytest.mark.asyncio
async def test_benchmark_corrections_overlay_substitutes_ground_truth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Corrections overlay replaces ground truths before scoring."""
    provider = BenchmarkProvider()
    _install_stub_client(monkeypatch, provider)

    # The stub answers "yellow" for the lamp question, but ground truth is "green".
    # Correction changes ground truth to "yellow" so it should now score as correct.
    corrections_path = tmp_path / "corrections.json"
    corrections_path.write_text(
        json.dumps({
            "conv-test-1:q3": {
                "original_ground_truth": "green",
                "corrected_ground_truth": "yellow",
                "reason": "Test correction",
            }
        }),
        encoding="utf-8",
    )
    benchmark = LoCoMoBenchmark(
        data_path=_write_dataset(tmp_path),
        llm_provider="openai",
        llm_api_key="test-openai-key",
        llm_model="answer-model",
        judge_model="judge-model",
        manifests_dir=MANIFESTS_DIR,
        corrections_path=corrections_path,
    )

    report = await benchmark.run()

    # Without corrections: q1=correct, q2=correct, q3=wrong (green vs yellow) → 2/3
    # With correction on q3: ground truth becomes "yellow" → q3 now correct → 3/3
    assert report.total_correct == 3
    assert report.total_questions == 3
    assert report.overall_accuracy == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_benchmark_can_reuse_retained_ingestion_db_without_persisting_questions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = BenchmarkProvider()
    _install_stub_client(monkeypatch, provider)
    data_path = _write_dataset(tmp_path)
    db_dir = tmp_path / "benchmark-dbs"
    progress_statuses: list[str] = []
    original_write_progress = LoCoMoBenchmark._write_ingestion_progress

    def capture_progress(*args, **kwargs) -> None:
        progress_statuses.append(str(kwargs["status"]))
        original_write_progress(*args, **kwargs)

    monkeypatch.setattr(
        LoCoMoBenchmark,
        "_write_ingestion_progress",
        staticmethod(capture_progress),
    )
    benchmark = LoCoMoBenchmark(
        data_path=data_path,
        llm_provider="openai",
        llm_api_key="test-openai-key",
        llm_model="answer-model",
        judge_model="judge-model",
        manifests_dir=MANIFESTS_DIR,
    )

    ingest_report = await benchmark.run(
        conversation_ids=["conv-test-1"],
        benchmark_db_dir=db_dir,
        ingest_only=True,
    )

    assert ingest_report.total_questions == 0
    retained_dbs = list(db_dir.glob("*/benchmark.db"))
    assert len(retained_dbs) == 1
    retained_db = retained_dbs[0]
    metadata = json.loads((retained_db.parent / "run_metadata.json").read_text())
    assert metadata["conversation_id"] == "conv-test-1"
    assert metadata["turn_count"] == 6
    assert metadata["migrations_sha256"]
    assert metadata["latest_migration_version"] == max(metadata["migration_versions"])
    assert metadata["latest_migration_version"] >= 1
    progress = json.loads((retained_db.parent / "ingestion_progress.json").read_text())
    assert progress["status"] == "workers_drained"
    assert "draining_workers" in progress_statuses
    assert progress_statuses.index("draining_workers") < progress_statuses.index("workers_drained")
    assert progress["conversation_id"] == "conv-test-1"
    assert progress["total_turns"] == 6
    assert progress["selected_turns"] == 6
    assert progress["ingested_turns"] == 6
    assert progress["last_turn_id"] == "D3:2"
    assert sum(
        1
        for request in provider.requests
        if request.metadata.get("purpose") == "memory_extraction"
    ) == 6

    eval_provider = BenchmarkProvider()
    _install_stub_client(monkeypatch, eval_provider)
    eval_benchmark = LoCoMoBenchmark(
        data_path=data_path,
        llm_provider="openai",
        llm_api_key="test-openai-key",
        llm_model="answer-model",
        judge_model="judge-model",
        manifests_dir=MANIFESTS_DIR,
    )

    report = await eval_benchmark.run(
        conversation_ids=["conv-test-1"],
        categories=[1],
        max_questions=1,
        reuse_db=retained_db,
    )

    assert report.total_questions == 1
    assert report.total_correct == 1
    assert report.model_info["benchmark_db"]["evaluate_only"] is True
    assert sum(
        1
        for request in eval_provider.requests
        if request.metadata.get("purpose") == "memory_extraction"
    ) == 0

    async with Atagia(
        db_path=retained_db,
        manifests_dir=MANIFESTS_DIR,
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/answer-model",
    ) as engine:
        runtime = engine.runtime
        assert runtime is not None
        connection = await runtime.open_connection()
        try:
            messages = await MessageRepository(
                connection,
                runtime.clock,
            ).list_messages_for_conversation("conv-test-1", "benchmark-user")
        finally:
            await connection.close()

    assert len(messages) == 6
    assert all("What color notebooks" not in str(message["text"]) for message in messages)


@pytest.mark.asyncio
async def test_benchmark_reuse_db_rejects_metadata_conversation_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = BenchmarkProvider()
    _install_stub_client(monkeypatch, provider)
    data_path = _write_two_conversation_dataset(tmp_path)
    db_dir = tmp_path / "benchmark-dbs"
    benchmark = LoCoMoBenchmark(
        data_path=data_path,
        llm_provider="openai",
        llm_api_key="test-openai-key",
        llm_model="answer-model",
        judge_model="judge-model",
        manifests_dir=MANIFESTS_DIR,
    )

    await benchmark.run(
        conversation_ids=["conv-test-1"],
        benchmark_db_dir=db_dir,
        ingest_only=True,
    )
    retained_db = next(db_dir.glob("*/benchmark.db"))

    with pytest.raises(ValueError, match="conversation mismatch"):
        await benchmark.run(
            conversation_ids=["conv-test-2"],
            categories=[1],
            max_questions=1,
            reuse_db=retained_db,
        )


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
            "warning_counts": {"failed_questions": 1, "tracebacks": 0},
            "retrieval_custody_summary": {
                "candidate_count": 2,
                "selected_count": 1,
                "channel_counts": {"fts": 2},
                "selected_channel_counts": {"fts": 1},
                "candidate_kind_counts": {"memory": 2},
                "composer_decision_counts": {"selected": 1, "rejected": 1},
                "filter_reason_counts": {"low_score": 1},
            },
        },
        duration_seconds=83.0,
    )
    summary = _format_report_summary(
        report=report,
        report_path=tmp_path / "report.json",
        manifest_path=tmp_path / "locomo-run-manifest.json",
        custody_path=tmp_path / "locomo-failed-custody.json",
    )

    assert "LoCoMo Benchmark Results" in summary
    assert "Overall accuracy: 50.0% (1/2)" in summary
    assert "Duration: 1m 23s" in summary
    assert "Model: openai / answer-model" in summary
    assert "Warning counts: failed_questions=1" in summary
    assert "Retrieval custody: candidates=2 selected=1" in summary
    assert "channels=fts=2" in summary
    assert "Cat 1 (single-hop):" in summary
    assert "Per-conversation:" in summary
    assert "conv-test-1:" in summary
    assert "Report saved to:" in summary
    assert "Run manifest saved to:" in summary
    assert "Failed-question custody saved to:" in summary


def test_build_run_manifest_includes_reproducibility_fields(tmp_path: Path) -> None:
    data_path = _write_dataset(tmp_path)
    benchmark = LoCoMoBenchmark(
        data_path=data_path,
        llm_provider="openai",
        llm_api_key="test-openai-key",
        llm_model="answer-model",
        judge_model="judge-model",
        manifests_dir=MANIFESTS_DIR,
    )
    report = BenchmarkReport(
        benchmark_name="LoCoMo",
        overall_accuracy=1.0,
        category_breakdown={1: 1.0},
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
                        memories_used=2,
                        retrieval_time_ms=42.0,
                    )
                ],
                accuracy=1.0,
                category_breakdown={1: 1.0},
                metadata={"benchmark_db_path": str(tmp_path / "benchmark.db")},
            )
        ],
        total_questions=1,
        total_correct=1,
        ablation_config=None,
        timestamp="2026-04-01T00:00:00+00:00",
        model_info={
            "provider": "openai",
            "warning_counts": {"failed_questions": 0},
            "retrieval_custody_summary": {
                "candidate_count": 1,
                "selected_count": 1,
            },
        },
        duration_seconds=1.0,
    )

    report_path = tmp_path / "locomo-report.json"
    report_payload = b'{"benchmark_name": "LoCoMo"}'
    report_path.write_bytes(report_payload)
    custody_path = tmp_path / "locomo-failed-custody.json"
    custody_payload = b'{"total_failed_questions": 0}'
    custody_path.write_bytes(custody_payload)
    taxonomy_path = tmp_path / "locomo-failure-taxonomy.json"
    taxonomy_payload = b'{"total_failed_questions": 0}'
    taxonomy_path.write_bytes(taxonomy_payload)

    manifest = benchmark.build_run_manifest(
        report,
        report_path=report_path,
        checkpoint_path=tmp_path / "checkpoint.json",
        custody_path=custody_path,
        taxonomy_path=taxonomy_path,
        failure_taxonomy_summary={"taxonomy_counts": {}},
    )

    assert manifest["dataset"]["sha256"]
    assert manifest["migrations"]["sha256"]
    assert manifest["migrations"]["latest_version"] == max(manifest["migrations"]["versions"])
    assert manifest["git"]["commit"]
    assert manifest["report_sha256"] == hashlib.sha256(report_payload).hexdigest()
    assert manifest["custody_path"] == str(custody_path)
    assert manifest["custody_sha256"] == hashlib.sha256(custody_payload).hexdigest()
    assert manifest["taxonomy_path"] == str(taxonomy_path)
    assert manifest["taxonomy_sha256"] == hashlib.sha256(taxonomy_payload).hexdigest()
    assert manifest["failure_taxonomy_summary"] == {"taxonomy_counts": {}}
    assert manifest["checkpoint_sha256"] is None
    assert manifest["benchmark_questions_persisted_as_messages"] is False
    assert manifest["partial_checkpoint_paths"] == []
    assert manifest["retained_db_paths"] == [str(tmp_path / "benchmark.db")]
    assert manifest["retrieval_custody_summary"] == {
        "candidate_count": 1,
        "selected_count": 1,
    }
    assert manifest["selection"] == {
        "conversation_ids": ["conv-test-1"],
        "selected_conversation_count": 1,
        "completed_question_count": 1,
    }
    assert manifest["result_summary"]["retrieval_time_ms"] == {
        "count": 1,
        "mean": 42.0,
        "min": 42.0,
        "max": 42.0,
    }
    assert manifest["result_summary"]["memories_used"] == {
        "count": 1,
        "mean": 2.0,
        "min": 2.0,
        "max": 2.0,
    }


def test_build_run_manifest_includes_parallel_checkpoint_paths(tmp_path: Path) -> None:
    data_path = _write_two_conversation_dataset(tmp_path)
    benchmark = LoCoMoBenchmark(
        data_path=data_path,
        llm_provider="openai",
        llm_api_key="test-openai-key",
        llm_model="answer-model",
        judge_model="judge-model",
        manifests_dir=MANIFESTS_DIR,
    )
    report = BenchmarkReport(
        benchmark_name="LoCoMo",
        overall_accuracy=1.0,
        category_breakdown={1: 1.0},
        conversations=[
            ConversationReport(
                conversation_id="conv-test-1",
                results=[],
                accuracy=0.0,
                category_breakdown={},
                metadata={},
            ),
            ConversationReport(
                conversation_id="conv/test 2",
                results=[],
                accuracy=0.0,
                category_breakdown={},
                metadata={},
            ),
        ],
        total_questions=0,
        total_correct=0,
        ablation_config=None,
        timestamp="2026-04-01T00:00:00+00:00",
        model_info={"provider": "openai", "parallel_conversations": 2},
        duration_seconds=1.0,
    )

    first_partial = tmp_path / "locomo-checkpoint-conv-test-1.json"
    first_partial_payload = b'{"conversation_id": "conv-test-1"}'
    first_partial.write_bytes(first_partial_payload)

    manifest = benchmark.build_run_manifest(
        report,
        report_path=tmp_path / "locomo-report.json",
        checkpoint_path=tmp_path / "locomo-checkpoint.json",
    )

    assert manifest["partial_checkpoint_paths"] == [
        {
            "conversation_id": "conv-test-1",
            "path": str(tmp_path / "locomo-checkpoint-conv-test-1.json"),
            "sha256": hashlib.sha256(first_partial_payload).hexdigest(),
        },
        {
            "conversation_id": "conv/test 2",
            "path": str(tmp_path / "locomo-checkpoint-conv-test-2.json"),
            "sha256": None,
        },
    ]


def test_format_benchmark_db_list_finds_nested_metadata(tmp_path: Path) -> None:
    metadata_dir = tmp_path / "group" / "locomo_conv-test_20260425T000000Z"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "benchmark.db").write_bytes(b"sqlite placeholder")
    (metadata_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "created_at": "2026-04-25T00:00:00+00:00",
                "conversation_id": "conv-test",
                "turn_count": 6,
                "llm_model": "answer-model",
            }
        ),
        encoding="utf-8",
    )

    summary = _format_benchmark_db_list(tmp_path)

    assert "timestamp | conversation_id | turns | model | status | messages | memories | summaries | topics | updated | db_path" in summary
    assert "conv-test" in summary
    assert "metadata_complete" in summary
    assert str(metadata_dir / "benchmark.db") in summary
    assert "Totals: entries=1 dbs=1" in summary
    assert "messages=0" in summary


def test_format_benchmark_db_list_includes_progress_only_snapshots(tmp_path: Path) -> None:
    progress_dir = tmp_path / "group" / "locomo_conv-progress_20260425T000000Z"
    progress_dir.mkdir(parents=True)
    (progress_dir / "benchmark.db").write_bytes(b"sqlite placeholder")
    (progress_dir / "ingestion_progress.json").write_text(
        json.dumps(
            {
                "status": "ingesting",
                "conversation_id": "conv-progress",
                "selected_turns": 10,
                "ingested_turns": 5,
                "updated_at": "2026-04-25T00:00:01+00:00",
            }
        ),
        encoding="utf-8",
    )

    summary = _format_benchmark_db_list(tmp_path)

    assert "conv-progress" in summary
    assert "5/10" in summary
    assert "ingesting" in summary
    assert str(progress_dir / "benchmark.db") in summary


def test_format_benchmark_db_list_json_includes_progress_metadata(tmp_path: Path) -> None:
    metadata_dir = tmp_path / "locomo_conv-test_20260425T000000Z"
    metadata_dir.mkdir()
    (metadata_dir / "benchmark.db").write_bytes(b"sqlite placeholder")
    (metadata_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "created_at": "2026-04-25T00:00:00+00:00",
                "conversation_id": "conv-test",
                "turn_count": 8,
                "llm_model": "answer-model",
            }
        ),
        encoding="utf-8",
    )
    (metadata_dir / "ingestion_progress.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "conversation_id": "conv-test",
                "selected_turns": 8,
                "ingested_turns": 8,
                "updated_at": "2026-04-25T00:00:02+00:00",
            }
        ),
        encoding="utf-8",
    )

    payload = json.loads(_format_benchmark_db_list_json(tmp_path))

    assert payload["db_dir"] == str(tmp_path)
    assert payload["exists"] is True
    assert payload["count"] == 1
    assert payload["totals"]["entry_count"] == 1
    assert payload["totals"]["has_db_count"] == 1
    assert payload["totals"]["message_count"] == 0
    assert isinstance(payload["totals"]["latest_file_updated_at"], str)
    entry = payload["entries"][0]
    assert isinstance(entry.pop("file_updated_at"), str)
    metadata_sha256 = hashlib.sha256(
        (metadata_dir / "run_metadata.json").read_bytes()
    ).hexdigest()
    progress_sha256 = hashlib.sha256(
        (metadata_dir / "ingestion_progress.json").read_bytes()
    ).hexdigest()
    assert entry == {
        "source": "metadata",
        "timestamp": "2026-04-25T00:00:00+00:00",
        "conversation_id": "conv-test",
        "turns": "8/8",
        "turn_count": 8,
        "selected_turns": 8,
        "ingested_turns": 8,
        "model": "answer-model",
        "status": "complete",
        "db_path": str(metadata_dir / "benchmark.db"),
        "metadata_path": str(metadata_dir / "run_metadata.json"),
        "metadata_sha256": metadata_sha256,
        "progress_path": str(metadata_dir / "ingestion_progress.json"),
        "progress_sha256": progress_sha256,
        "has_db": True,
        "db_bytes": len(b"sqlite placeholder"),
        "wal_bytes": 0,
        "shm_bytes": 0,
        "total_bytes": len(b"sqlite placeholder"),
        "message_count": None,
        "memory_object_count": None,
        "memory_embedding_metadata_count": None,
        "summary_view_count": None,
        "retrieval_event_count": None,
        "artifact_count": None,
        "artifact_chunk_count": None,
        "conversation_topic_count": None,
        "conversation_topic_event_count": None,
        "conversation_topic_source_count": None,
    }


def test_format_benchmark_db_list_json_includes_db_only_snapshots(tmp_path: Path) -> None:
    db_dir = tmp_path / "locomo_conv-42_20260426T061904Z"
    db_dir.mkdir()
    (db_dir / "benchmark.db").write_bytes(b"sqlite placeholder")
    (db_dir / "benchmark.db-wal").write_bytes(b"wal bytes")

    payload = json.loads(_format_benchmark_db_list_json(tmp_path))

    assert payload["count"] == 1
    entry = payload["entries"][0]
    assert isinstance(entry.pop("file_updated_at"), str)
    assert entry == {
        "source": "db",
        "timestamp": "20260426T061904Z",
        "conversation_id": "conv-42",
        "turns": "",
        "turn_count": None,
        "selected_turns": None,
        "ingested_turns": None,
        "model": "",
        "status": "db_present",
        "db_path": str(db_dir / "benchmark.db"),
        "metadata_path": None,
        "metadata_sha256": None,
        "progress_path": None,
        "progress_sha256": None,
        "has_db": True,
        "db_bytes": len(b"sqlite placeholder"),
        "wal_bytes": len(b"wal bytes"),
        "shm_bytes": 0,
        "total_bytes": len(b"sqlite placeholder") + len(b"wal bytes"),
        "message_count": None,
        "memory_object_count": None,
        "memory_embedding_metadata_count": None,
        "summary_view_count": None,
        "retrieval_event_count": None,
        "artifact_count": None,
        "artifact_chunk_count": None,
        "conversation_topic_count": None,
        "conversation_topic_event_count": None,
        "conversation_topic_source_count": None,
    }


def test_format_benchmark_db_list_json_counts_sqlite_rows(tmp_path: Path) -> None:
    db_dir = tmp_path / "locomo_conv-42_20260426T061904Z"
    db_dir.mkdir()
    db_path = db_dir / "benchmark.db"
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("CREATE TABLE messages (id TEXT PRIMARY KEY)")
        connection.execute("CREATE TABLE memory_objects (id TEXT PRIMARY KEY)")
        connection.execute("CREATE TABLE memory_embedding_metadata (memory_id TEXT PRIMARY KEY)")
        connection.execute("CREATE TABLE summary_views (id TEXT PRIMARY KEY)")
        connection.execute("CREATE TABLE retrieval_events (id TEXT PRIMARY KEY)")
        connection.execute("CREATE TABLE artifacts (id TEXT PRIMARY KEY)")
        connection.execute("CREATE TABLE artifact_chunks (id TEXT PRIMARY KEY)")
        connection.execute("CREATE TABLE conversation_topics (id TEXT PRIMARY KEY)")
        connection.execute("CREATE TABLE conversation_topic_events (id TEXT PRIMARY KEY)")
        connection.execute("CREATE TABLE conversation_topic_sources (id TEXT PRIMARY KEY)")
        connection.executemany("INSERT INTO messages (id) VALUES (?)", [("msg_1",), ("msg_2",)])
        connection.execute("INSERT INTO memory_objects (id) VALUES (?)", ("mem_1",))
        connection.execute("INSERT INTO memory_embedding_metadata (memory_id) VALUES (?)", ("mem_1",))
        connection.executemany(
            "INSERT INTO summary_views (id) VALUES (?)",
            [("sum_1",), ("sum_2",), ("sum_3",)],
        )
        connection.execute("INSERT INTO retrieval_events (id) VALUES (?)", ("evt_1",))
        connection.execute("INSERT INTO artifacts (id) VALUES (?)", ("art_1",))
        connection.executemany(
            "INSERT INTO artifact_chunks (id) VALUES (?)",
            [("chk_1",), ("chk_2",)],
        )
        connection.execute("INSERT INTO conversation_topics (id) VALUES (?)", ("tpc_1",))
        connection.executemany(
            "INSERT INTO conversation_topic_events (id) VALUES (?)",
            [("evt_topic_1",), ("evt_topic_2",)],
        )
        connection.executemany(
            "INSERT INTO conversation_topic_sources (id) VALUES (?)",
            [("src_1",), ("src_2",), ("src_3",)],
        )
        connection.commit()
    finally:
        connection.close()

    payload = json.loads(_format_benchmark_db_list_json(tmp_path))
    summary = _format_benchmark_db_list(tmp_path)

    assert payload["count"] == 1
    assert payload["entries"][0]["message_count"] == 2
    assert payload["entries"][0]["memory_object_count"] == 1
    assert payload["entries"][0]["memory_embedding_metadata_count"] == 1
    assert payload["entries"][0]["summary_view_count"] == 3
    assert payload["entries"][0]["retrieval_event_count"] == 1
    assert payload["entries"][0]["artifact_count"] == 1
    assert payload["entries"][0]["artifact_chunk_count"] == 2
    assert payload["entries"][0]["conversation_topic_count"] == 1
    assert payload["entries"][0]["conversation_topic_event_count"] == 2
    assert payload["entries"][0]["conversation_topic_source_count"] == 3
    assert payload["totals"]["message_count"] == 2
    assert payload["totals"]["memory_object_count"] == 1
    assert payload["totals"]["memory_embedding_metadata_count"] == 1
    assert payload["totals"]["summary_view_count"] == 3
    assert payload["totals"]["retrieval_event_count"] == 1
    assert payload["totals"]["artifact_count"] == 1
    assert payload["totals"]["artifact_chunk_count"] == 2
    assert payload["totals"]["conversation_topic_count"] == 1
    assert payload["totals"]["conversation_topic_event_count"] == 2
    assert payload["totals"]["conversation_topic_source_count"] == 3
    assert "Totals: entries=1 dbs=1" in summary
    assert "latest_update=" in summary
    assert "messages=2" in summary
    assert "memory_objects=1" in summary
    assert "embedding_metadata=1" in summary
    assert "summary_views=3" in summary
    assert "retrieval_events=1" in summary
    assert "artifacts=1" in summary
    assert "artifact_chunks=2" in summary
    assert "conversation_topics=1" in summary
    assert "topic_events=2" in summary
    assert "topic_sources=3" in summary


def test_format_benchmark_db_snapshot_diff_summarizes_growth(tmp_path: Path) -> None:
    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"
    before_path.write_text(
        json.dumps(
            {
                "count": 1,
                "totals": {
                    "latest_file_updated_at": "2026-04-26T10:00:00+00:00",
                    "total_bytes": 10,
                    "message_count": 2,
                    "memory_object_count": 1,
                    "summary_view_count": 0,
                },
                "entries": [
                    {
                        "db_path": "db-a/benchmark.db",
                        "conversation_id": "conv-a",
                        "status": "ingesting",
                        "file_updated_at": "2026-04-26T10:00:00+00:00",
                        "total_bytes": 10,
                        "message_count": 2,
                        "memory_object_count": 1,
                        "summary_view_count": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    after_path.write_text(
        json.dumps(
            {
                "count": 2,
                "totals": {
                    "latest_file_updated_at": "2026-04-26T10:05:00+00:00",
                    "total_bytes": 29,
                    "message_count": 5,
                    "memory_object_count": 3,
                    "summary_view_count": 1,
                },
                "entries": [
                    {
                        "db_path": "db-a/benchmark.db",
                        "conversation_id": "conv-a",
                        "status": "workers_drained",
                        "file_updated_at": "2026-04-26T10:05:00+00:00",
                        "total_bytes": 22,
                        "message_count": 5,
                        "memory_object_count": 3,
                        "summary_view_count": 1,
                    },
                    {
                        "db_path": "db-b/benchmark.db",
                        "conversation_id": "conv-b",
                        "status": "db_present",
                        "file_updated_at": "2026-04-26T10:04:00+00:00",
                        "total_bytes": 7,
                        "message_count": 0,
                        "memory_object_count": 0,
                        "summary_view_count": 0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = json.loads(
        _format_benchmark_db_snapshot_diff_json(before_path, after_path)
    )
    summary = _format_benchmark_db_snapshot_diff(before_path, after_path)

    assert payload["before_count"] == 1
    assert payload["after_count"] == 2
    assert payload["new_db_paths"] == ["db-b/benchmark.db"]
    assert payload["totals_delta"]["message_count"] == 3
    assert payload["totals_delta"]["memory_object_count"] == 2
    assert payload["totals_delta"]["summary_view_count"] == 1
    assert payload["entries"][0]["deltas"]["total_bytes"] == 12
    assert "Entries: 1 -> 2" in summary
    assert "message_count=+3" in summary
    assert "New DBs: db-b/benchmark.db" in summary
    assert "conv-a: messages=+3 memories=+2 summaries=+1 bytes=+12" in summary


def test_format_run_log_summary_counts_common_failure_lines(tmp_path: Path) -> None:
    log_path = tmp_path / "run.log"
    log_path.write_text(
        "\n".join(
            [
                "Ingesting 10/20 turns for conv-1...",
                "Ingested 5/10 turns for conv-1.",
                "Ingested 6/10 turns for conv-1.",
                "Ingested 2/8 turns for conv-2.",
                "Conversation 1/1 (conv-1): question 1/2",
                "Failed to process contract job stm_1",
                "Consequence tendency inference fallback to chain without tendency",
                "Traceback (most recent call last):",
                "atagia.services.llm_client.StructuredOutputError: Provider returned invalid structured output",
                "pydantic_core._pydantic_core.ValidationError: 1 validation error for _TendencyResult",
                "memory_statement",
                "  Extra inputs are not permitted [type=extra_forbidden]",
                "Provider rate limit HTTP 429",
            ]
        ),
        encoding="utf-8",
    )

    payload = json.loads(_format_run_log_summary_json(log_path))
    summary = _format_run_log_summary(log_path)

    assert payload["exists"] is True
    assert isinstance(payload["generated_at"], str)
    assert isinstance(payload["seconds_since_update"], float)
    assert payload["line_count"] == 13
    assert payload["counts"]["ingestion_started_lines"] == 1
    assert payload["counts"]["turn_progress_lines"] == 3
    assert payload["counts"]["question_started_lines"] == 1
    assert payload["counts"]["consequence_tendency_fallback_lines"] == 1
    assert payload["counts"]["failed_worker_jobs"] == 1
    assert payload["counts"]["contract_worker_failures"] == 1
    assert payload["counts"]["tracebacks"] == 1
    assert payload["counts"]["structured_output_errors"] == 1
    assert payload["counts"]["validation_errors"] == 1
    assert payload["counts"]["provider_rate_limit_lines"] == 1
    assert payload["counts"]["provider_status_error_lines"] == 1
    assert payload["validation_error_schemas"] == {"_TendencyResult": 1}
    assert payload["validation_error_fields"] == {"memory_statement": 1}
    assert payload["latest_lines"] == {
        "consequence_tendency_fallback": (
            "Consequence tendency inference fallback to chain without tendency"
        ),
        "contract_worker_failures": "Failed to process contract job stm_1",
        "failed_worker_job": "Failed to process contract job stm_1",
        "ingestion_started": "Ingesting 10/20 turns for conv-1...",
        "provider_rate_limit": "Provider rate limit HTTP 429",
        "provider_status_error": "Provider rate limit HTTP 429",
        "question_started": "Conversation 1/1 (conv-1): question 1/2",
        "structured_output_error": (
            "atagia.services.llm_client.StructuredOutputError: "
            "Provider returned invalid structured output"
        ),
        "traceback": "Traceback (most recent call last):",
        "validation_error": (
            "pydantic_core._pydantic_core.ValidationError: "
            "1 validation error for _TendencyResult"
        ),
        "turn_progress": "Ingested 2/8 turns for conv-2.",
    }
    assert payload["ingestion_started_by_conversation"] == {
        "conv-1": {
            "line": "Ingesting 10/20 turns for conv-1...",
            "selected_turns": 10,
            "source_turns": 20,
        }
    }
    assert payload["turn_progress_by_conversation"] == {
        "conv-1": {
            "line": "Ingested 6/10 turns for conv-1.",
            "completed_turns": 6,
            "total_turns": 10,
        },
        "conv-2": {
            "line": "Ingested 2/8 turns for conv-2.",
            "completed_turns": 2,
            "total_turns": 8,
        },
    }
    assert payload["turn_progress_totals"] == {
        "conversation_count": 2,
        "conversations_with_progress": 2,
        "completed_conversations": 0,
        "completed_turns": 8,
        "total_turns": 18,
        "remaining_turns": 10,
        "completion_ratio": 0.444444,
    }
    assert payload["question_progress_by_conversation"] == {
        "conv-1": {
            "line": "Conversation 1/1 (conv-1): question 1/2",
            "current_question": 1,
            "total_questions": 2,
        }
    }
    assert "failed_worker_jobs=1" in summary
    assert "Validation schemas: _TendencyResult=1" in summary
    assert "Validation fields: memory_statement=1" in summary
    assert "age_seconds=" in summary
    assert "question_started: Conversation 1/1 (conv-1): question 1/2" in summary
    assert "Ingestion started by conversation: conv-1=10/20" in summary
    assert "Turn progress by conversation: conv-1=6/10 conv-2=2/8" in summary
    assert (
        "Turn progress totals: 8/18 (44.4%) remaining=10 "
        "conversations=2 with_progress=2 complete=0"
    ) in summary
    assert "Question progress by conversation: conv-1=1/2" in summary
