"""TypeSafe System One adapter for explicit choices and five-level scores.

HTTP contract: https://docs.typesafe.ai/api. Jev does not generate chat text.
The shared LLM client still owns call metering, run guards, and retry policy.
"""

from __future__ import annotations

import math
from typing import AsyncIterator

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from atagia.models.schemas_decisions import (
    ChoiceAnswer,
    ChoiceQuestion,
    ScoreAnswer,
    ScoreQuestion,
)
from atagia.diagnostics.recorder import capture_raw_response, capture_sent_payload
from atagia.services.llm_client import (
    ConfigurationError,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMError,
    LLMProvider,
    LLMRequestError,
    LLMStreamEvent,
    TransientLLMError,
    retry_after_seconds_from_headers,
)


class _Usage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    input_tokens: int = Field(ge=0, strict=True)
    output_tokens: int = Field(ge=0, strict=True)


class _EvaluationResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = Field(min_length=1)
    answers: dict[str, ChoiceAnswer | ScoreAnswer]
    usage: _Usage


def _is_rounding_consistent_2dp(probabilities: dict[str, float]) -> bool:
    """Allow a sum discrepancy only when two-decimal transport rounding explains it."""
    values = tuple(probabilities.values())
    if sum(values) <= 0 or any(
        not math.isclose(value * 100, round(value * 100), rel_tol=0, abs_tol=1e-8)
        for value in values
    ):
        return False
    lower = sum(max(0.0, value - 0.005) for value in values)
    upper = sum(min(1.0, value + 0.005) for value in values)
    return lower <= 1.0 + 1e-9 and upper >= 1.0 - 1e-9


class TypeSafeProvider(LLMProvider):
    """Native typed choices and five-level scores, with no chat fallback."""

    name = "typesafe"
    supports_embeddings = False
    supports_native_structured_output = False
    supports_choices = True
    supports_scores = True

    def __init__(
        self,
        api_key: str,
        *,
        request_timeout_seconds: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key.strip():
            raise ConfigurationError("ATAGIA_TYPESAFE_API_KEY is required")
        self._api_key = api_key
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=request_timeout_seconds,
            follow_redirects=False,
            trust_env=False,
        )

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        choices = request.choice_questions
        scores = request.score_questions
        if bool(choices) == bool(scores):
            raise ConfigurationError(
                "TypeSafe requires either choice_questions or score_questions, not chat text"
            )
        questions = choices or scores
        assert questions is not None
        if request.tools or request.response_schema or request.include_thinking:
            raise ConfigurationError("TypeSafe decisions do not support tools, JSON schemas, or thinking")
        if "," in request.model:
            raise ConfigurationError("TypeSafe does not accept thinking-level suffixes")

        payload = {
            "model": request.model,
            "state": [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ],
            "questions": {
                key: question.model_dump(mode="json")
                for key, question in questions.items()
            },
        }
        capture_sent_payload(payload)
        try:
            response = await self._client.post(
                "https://api.typesafe.ai/v1/systemone",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
            )
        except httpx.TransportError as exc:
            # Do not echo provider bodies, request content, or credentials.
            raise TransientLLMError(f"TypeSafe transport failed ({type(exc).__name__})") from None
        if response.status_code in {429, 500, 502, 503, 504, 529}:
            raise TransientLLMError(
                f"TypeSafe HTTP {response.status_code}",
                retry_after_seconds=retry_after_seconds_from_headers(response.headers),
            )
        if not response.is_success:
            raise LLMRequestError(
                f"TypeSafe HTTP {response.status_code}", status_code=response.status_code
            )
        capture_raw_response(response.text)
        try:
            result = _EvaluationResponse.model_validate(response.json())
        except (ValidationError, ValueError):
            raise LLMError("TypeSafe returned an invalid evaluation response") from None
        if result.answers.keys() != questions.keys():
            raise LLMError("TypeSafe answer IDs do not match the requested questions")
        distribution_diagnostics: dict[str, dict[str, float | str]] = {}
        for key, answer in result.answers.items():
            question = questions[key]
            if isinstance(question, ScoreQuestion):
                if not isinstance(answer, ScoreAnswer):
                    raise LLMError("TypeSafe answer type does not match the score question")
                levels = {str(index): description for index, description in enumerate(question.criteria)}
                probabilities = answer.probabilities
                if answer.legend != levels or probabilities.keys() != levels.keys():
                    raise LLMError("TypeSafe returned an invalid score legend or distribution")
                if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()):
                    raise LLMError("TypeSafe returned an invalid score distribution")
                probability_sum = sum(probabilities.values())
                rounded = _is_rounding_consistent_2dp(probabilities)
                if not math.isclose(probability_sum, 1.0, abs_tol=0.002) and not rounded:
                    raise LLMError("TypeSafe returned an invalid score distribution")
                weighted = sum(index * probabilities[str(index)] for index in range(5))
                # Preserve the provider's weighted answer. Only the same
                # two-decimal transport envelope accepted for Choice can
                # account for a difference from exposed probabilities.
                score_is_2dp = math.isclose(answer.score * 100, round(answer.score * 100), abs_tol=1e-8)
                tolerance = 0.055 if rounded and score_is_2dp else 1e-6
                if not math.isclose(answer.score, weighted, abs_tol=tolerance):
                    raise LLMError("TypeSafe returned an inconsistent weighted score")
                if not math.isclose(probability_sum, 1.0, abs_tol=0.002) or not math.isclose(
                    answer.score, weighted, abs_tol=1e-6
                ):
                    distribution_diagnostics[key] = {
                        "original_sum": probability_sum,
                        "weighted_from_probabilities": weighted,
                        "reported_score": answer.score,
                        "precision_criterion": "rounding_consistent_2dp",
                    }
                continue
            if not isinstance(question, ChoiceQuestion) or not isinstance(answer, ChoiceAnswer):
                raise LLMError("TypeSafe answer type does not match the choice question")
            options = question.criteria
            probabilities = answer.probabilities
            if (
                answer.choice not in options
                or probabilities.keys() != options.keys()
                or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values())
                or probabilities[answer.choice] < max(probabilities.values()) - 1e-6
            ):
                raise LLMError("TypeSafe returned an invalid choice distribution")
            probability_sum = sum(probabilities.values())
            if not math.isclose(probability_sum, 1.0, abs_tol=0.002):
                if not _is_rounding_consistent_2dp(probabilities):
                    raise LLMError("TypeSafe returned an invalid choice distribution")
                distribution_diagnostics[key] = {
                    "original_sum": probability_sum,
                    "precision_criterion": "rounding_consistent_2dp",
                }

        raw_response = result.model_dump(mode="json")
        if distribution_diagnostics:
            raw_response[
                "score_distribution_diagnostics" if scores else "choice_distribution_diagnostics"
            ] = distribution_diagnostics

        return LLMCompletionResponse(
            provider=self.name,
            model=result.model,
            choice_answers={key: answer for key, answer in result.answers.items() if isinstance(answer, ChoiceAnswer)},
            score_answers={key: answer for key, answer in result.answers.items() if isinstance(answer, ScoreAnswer)},
            usage=result.usage.model_dump(),
            finish_reason="stop",
            raw_response=raw_response,
        )

    async def stream(self, request: LLMCompletionRequest) -> AsyncIterator[LLMStreamEvent]:
        raise ConfigurationError("TypeSafe returns typed decisions; streaming text is unsupported")
        yield  # pragma: no cover - async iterator contract

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
