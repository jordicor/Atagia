"""Benchmark-only GLiNER2.5 adapter for one production finite-choice card.

The text is TypeSafe's ordered state array. The question ID, instructions,
option IDs, and descriptions go to GLiNER's native classification schema.
One request is one classification task, so distinct cards cannot be fused.
"""

from __future__ import annotations

import inspect
import json
import math
from time import perf_counter
from typing import Any, AsyncIterator
import warnings

from atagia.models.schemas_decisions import ChoiceAnswer
from atagia.services.llm_client import (
    ConfigurationError,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMProvider,
    LLMStreamEvent,
)


class _StrictProcessor:
    """Force processor failures to surface and count actual encoded tokens."""

    def __init__(self, processor: Any) -> None:
        if "error_policy" not in inspect.signature(processor.collate_fn_inference).parameters:
            raise ConfigurationError("GLiNER processor must support error_policy='raise'")
        self._processor = processor
        self.input_tokens: int | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._processor, name)

    def collate_fn_inference(self, rows: Any, *, max_len: int | None = None) -> Any:
        if max_len is not None:
            raise ConfigurationError("GLiNER benchmark must not truncate source text")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            batch = self._processor.collate_fn_inference(
                rows, max_len=None, error_policy="raise"
            )
        if len(rows) != 1:
            raise ConfigurationError("GLiNER benchmark requires one card per inference")
        self.input_tokens = int(batch.attention_mask[0].sum().item())
        if self.input_tokens <= 0:
            raise RuntimeError("GLiNER encoded no input tokens")
        return batch


def _request_input(
    request: LLMCompletionRequest,
) -> tuple[str, str, str, dict[str, str | None]]:
    questions = request.choice_questions
    if not questions or len(questions) != 1:
        raise ConfigurationError("GLiNER benchmark requires exactly one choice question")
    if request.tools or request.response_schema or request.include_thinking:
        raise ConfigurationError("GLiNER choices do not support tools, JSON schemas, or thinking")
    question_id, question = next(iter(questions.items()))
    if not question.criteria or len(question.criteria) < 2:
        raise ConfigurationError("GLiNER benchmark requires at least two choices")
    state = [
        {"role": message.role, "content": message.content}
        for message in request.messages
    ]
    return (
        json.dumps(state, ensure_ascii=False),
        question_id,
        question.instructions,
        dict(question.criteria),
    )


def _checked_answer(task: Any, labels: list[str]) -> ChoiceAnswer:
    if not isinstance(task, dict):
        raise RuntimeError("GLiNER returned no classification task")
    selected = task.get("value")
    probabilities = task.get("probabilities")
    confidence = task.get("confidence")
    if (
        not isinstance(selected, str)
        or not isinstance(probabilities, dict)
        or list(probabilities) != labels
        or selected not in probabilities
        or not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
    ):
        raise RuntimeError("GLiNER returned an invalid choice distribution")
    if any(
        not isinstance(p, (int, float))
        or isinstance(p, bool)
        or not math.isfinite(p)
        or not 0 <= p <= 1
        for p in probabilities.values()
    ):
        raise RuntimeError("GLiNER returned an invalid choice distribution")
    if (
        not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-5)
        or probabilities[selected] < max(probabilities.values()) - 1e-6
        or not math.isfinite(confidence)
        or not math.isclose(confidence, probabilities[selected], abs_tol=1e-5)
    ):
        raise RuntimeError("GLiNER returned an invalid choice distribution")
    return ChoiceAnswer(
        type="choice",
        choice=selected,
        probabilities=probabilities,
        confidence=confidence,
    )


class GLiNERDecisionProvider(LLMProvider):
    """Inject an already loaded local Classifier; never load or route a model."""

    name = "gliner"
    supports_embeddings = False
    supports_native_structured_output = False
    supports_choices = True

    def __init__(
        self,
        classifier: Any,
        *,
        model_id: str,
        expected_cuda_name: str | None = None,
    ) -> None:
        self._classifier = classifier
        self._model_id = model_id
        self._expected_cuda_name = expected_cuda_name
        self._processor = _StrictProcessor(classifier.scorer.processor)
        classifier.scorer.processor = self._processor

    def _synchronize(self) -> None:
        if self._expected_cuda_name is None:
            return
        import torch

        device = str(self._classifier.device)
        if (
            not device.startswith("cuda")
            or not torch.cuda.is_available()
            or torch.cuda.device_count() != 1
        ):
            raise ConfigurationError("GLiNER benchmark requires the configured CUDA device")
        index = int(device.split(":", 1)[1]) if ":" in device else torch.cuda.current_device()
        if index != 0:
            raise ConfigurationError("GLiNER benchmark requires the single visible CUDA device")
        if torch.cuda.get_device_name(index) != self._expected_cuda_name:
            raise ConfigurationError("GLiNER benchmark is running on the wrong CUDA device")
        try:
            parameter_device = str(next(self._classifier.model.parameters()).device)
        except (AttributeError, StopIteration) as exc:
            raise ConfigurationError("GLiNER model has no inspectable parameters") from exc
        if not parameter_device.startswith("cuda"):
            raise ConfigurationError("GLiNER model parameters are not on CUDA")
        torch.cuda.synchronize(index)

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        text, question_id, instructions, criteria = _request_input(request)
        from gliner2.classification import ClassificationConfig, ClassificationSchema

        schema = ClassificationSchema().single(
            question_id,
            criteria,
            instruction=instructions,
            activation="softmax",
        )
        config = ClassificationConfig(
            decoder="independent",
            include_confidence=True,
            max_len=None,
            on_infeasible="raise",
        )
        self._processor.input_tokens = None
        self._synchronize()
        started = perf_counter()
        result = self._classifier.classify(text, schema, config=config)
        self._synchronize()
        elapsed_ms = (perf_counter() - started) * 1000.0
        task = result.to_dict().get(question_id)
        answer = _checked_answer(task, list(criteria))
        if self._processor.input_tokens is None:
            raise RuntimeError("GLiNER inference bypassed the strict processor")
        return LLMCompletionResponse(
            provider=self.name,
            model=self._model_id,
            choice_answers={question_id: answer},
            usage={"input_tokens": self._processor.input_tokens, "output_tokens": 0},
            finish_reason="stop",
            raw_response={
                "elapsed_ms": elapsed_ms,
                "encoder_tokens": self._processor.input_tokens,
                "device": str(self._classifier.device),
                "dtype": str(self._classifier.dtype),
                "max_len": None,
            },
        )

    async def stream(self, request: LLMCompletionRequest) -> AsyncIterator[LLMStreamEvent]:
        raise ConfigurationError("GLiNER benchmark returns typed decisions, not text")
        yield  # pragma: no cover - async iterator contract
