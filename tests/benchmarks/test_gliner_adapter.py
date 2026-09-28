"""Offline contract tests for the benchmark-only GLiNER decision adapter."""

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from atagia.models.schemas_decisions import ChoiceQuestion
from atagia.services.llm_client import ConfigurationError, LLMCompletionRequest, LLMMessage
from benchmarks.local_decision_cards.gliner_adapter import GLiNERDecisionProvider


class _TokenCount:
    def sum(self) -> _TokenCount:
        return self

    def item(self) -> int:
        return 43


class _Processor:
    def __init__(self) -> None:
        self.last_error_policy: str | None = None
        self.last_max_len: int | None = 1

    def collate_fn_inference(
        self, rows: Any, *, max_len: int | None = None, error_policy: str = "fallback"
    ) -> Any:
        self.last_error_policy = error_policy
        self.last_max_len = max_len
        return SimpleNamespace(attention_mask=[_TokenCount()])


class _Classifier:
    def __init__(self, task: dict[str, Any] | None = None) -> None:
        self.device = "cpu"
        self.dtype = "torch.float32"
        self.processor = _Processor()
        self.scorer = SimpleNamespace(processor=self.processor)
        self.task = task or {
            "value": "link",
            "confidence": 0.8,
            "probabilities": {"link": 0.8, "other": 0.2},
        }
        self.text: str | None = None
        self.schema: Any = None
        self.config: Any = None

    def classify(self, text: str, schema: Any, *, config: Any) -> Any:
        self.text, self.schema, self.config = text, schema, config
        self.scorer.processor.collate_fn_inference([(text, schema)], max_len=config.max_len)
        return SimpleNamespace(to_dict=lambda: {schema.name: self.task})


@pytest.fixture(autouse=True)
def fake_gliner(monkeypatch: pytest.MonkeyPatch) -> None:
    package = ModuleType("gliner2")
    classification = ModuleType("gliner2.classification")

    class ClassificationSchema:
        def single(self, name: str, labels: dict[str, str | None], **kwargs: Any) -> Any:
            self.name, self.labels, self.kwargs = name, labels, kwargs
            return self

    class ClassificationConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    classification.ClassificationSchema = ClassificationSchema  # type: ignore[attr-defined]
    classification.ClassificationConfig = ClassificationConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gliner2", package)
    monkeypatch.setitem(sys.modules, "gliner2.classification", classification)


def _request() -> LLMCompletionRequest:
    return LLMCompletionRequest(
        model="gliner-test",
        messages=[
            LLMMessage(role="system", content="Use the full state (including history)."),
            LLMMessage(role="user", content="Esta opción enlaza con la anterior.", name="tester"),
        ],
        choice_questions={
            "consequence_link": ChoiceQuestion(
                instructions="Determine the prior link, if any.",
                criteria={
                    "link": "This refers to the earlier suggestion, not a new one.",
                    "other": None,
                },
            )
        },
    )


@pytest.mark.asyncio
async def test_preserves_complete_card_and_native_distribution() -> None:
    classifier = _Classifier()
    provider = GLiNERDecisionProvider(classifier, model_id="fastino/gliner2.5-multi-v1")

    response = await provider.complete(_request())

    assert json.loads(classifier.text) == [
        {"role": "system", "content": "Use the full state (including history)."},
        {"role": "user", "content": "Esta opción enlaza con la anterior."},
    ]
    assert classifier.schema.name == "consequence_link"
    assert classifier.schema.labels == {
        "link": "This refers to the earlier suggestion, not a new one.",
        "other": None,
    }
    assert classifier.schema.kwargs == {
        "instruction": "Determine the prior link, if any.",
        "activation": "softmax",
    }
    assert classifier.config.max_len is None
    assert classifier.processor.last_error_policy == "raise"
    assert classifier.processor.last_max_len is None
    assert response.choice_answers["consequence_link"].choice == "link"
    assert response.choice_answers["consequence_link"].probabilities == {
        "link": 0.8,
        "other": 0.2,
    }
    assert response.choice_answers["consequence_link"].confidence == 0.8
    assert response.usage == {"input_tokens": 43, "output_tokens": 0}
    assert response.raw_response["encoder_tokens"] == 43
    assert response.raw_response["dtype"] == "torch.float32"


@pytest.mark.asyncio
async def test_rejects_multiple_questions_without_running_model() -> None:
    classifier = _Classifier()
    provider = GLiNERDecisionProvider(classifier, model_id="test")
    request = _request()
    request.choice_questions["another"] = ChoiceQuestion(
        instructions="Separate card", criteria={"yes": None, "no": None}
    )

    with pytest.raises(ConfigurationError, match="exactly one"):
        await provider.complete(request)
    assert classifier.text is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "task",
    [
        {"value": "link", "confidence": 0.8, "probabilities": {"link": 0.8}},
        {"value": "link", "confidence": 0.8, "probabilities": {"link": 0.4, "other": 0.6}},
        {"value": "link", "confidence": 0.8, "probabilities": {"link": 0.6, "other": 0.2}},
        {"value": "link", "confidence": 0.8, "probabilities": {"link": float("nan"), "other": 0.2}},
        {"value": "link", "confidence": 0.9, "probabilities": {"link": 0.8, "other": 0.2}},
    ],
)
async def test_rejects_invalid_native_distribution(task: dict[str, Any]) -> None:
    provider = GLiNERDecisionProvider(_Classifier(task), model_id="test")

    with pytest.raises(RuntimeError, match="invalid choice distribution"):
        await provider.complete(_request())


@pytest.mark.asyncio
async def test_processor_failure_is_not_replaced_with_a_fallback_record() -> None:
    classifier = _Classifier()

    def fail_if_strict(
        rows: Any, *, max_len: int | None = None, error_policy: str = "fallback"
    ) -> Any:
        assert error_policy == "raise"
        raise ValueError("malformed record")

    classifier.processor.collate_fn_inference = fail_if_strict  # type: ignore[method-assign]
    provider = GLiNERDecisionProvider(classifier, model_id="test")

    with pytest.raises(ValueError, match="malformed record"):
        await provider.complete(_request())


@pytest.mark.asyncio
async def test_configured_gpu_rejects_cpu_before_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: True)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    classifier = _Classifier()
    provider = GLiNERDecisionProvider(
        classifier, model_id="test", expected_cuda_name="NVIDIA GeForce RTX 3090"
    )

    with pytest.raises(ConfigurationError, match="configured CUDA device"):
        await provider.complete(_request())
    assert classifier.text is None


@pytest.mark.asyncio
async def test_configured_gpu_requires_one_visible_matching_device_and_cuda_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synchronized: list[int] = []
    torch = ModuleType("torch")
    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_name=lambda index: "NVIDIA GeForce RTX 3090",
        synchronize=synchronized.append,
    )
    torch.cuda = cuda  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    classifier = _Classifier()
    classifier.device = "cuda:0"
    parameter = SimpleNamespace(device="cuda:0")
    classifier.model = SimpleNamespace(parameters=lambda: iter([parameter]))
    provider = GLiNERDecisionProvider(
        classifier, model_id="test", expected_cuda_name="NVIDIA GeForce RTX 3090"
    )

    response = await provider.complete(_request())
    assert response.raw_response["device"] == "cuda:0"
    assert synchronized == [0, 0]

    cuda.device_count = lambda: 2
    with pytest.raises(ConfigurationError, match="configured CUDA device"):
        await provider.complete(_request())

    cuda.device_count = lambda: 1
    parameter.device = "cpu"
    with pytest.raises(ConfigurationError, match="parameters are not on CUDA"):
        await provider.complete(_request())
