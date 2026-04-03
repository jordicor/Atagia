"""LLM-judge scoring for benchmark answers."""

from __future__ import annotations

import json
import re

from benchmarks.base import ScoreResult
from atagia.services.llm_client import LLMClient, LLMCompletionRequest, LLMMessage

_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
_DEFAULT_REASONING = "Judge response could not be parsed as valid verdict JSON."


class LLMJudgeScorer:
    """Score predictions using the configured Atagia LLM client."""

    def __init__(self, llm_client: LLMClient[object], judge_model: str) -> None:
        self._llm_client = llm_client
        self._judge_model = judge_model

    async def score(
        self,
        question: str,
        prediction: str,
        ground_truth: str,
    ) -> ScoreResult:
        """Return a binary judge verdict for one question-answer pair."""
        response = await self._llm_client.complete(
            LLMCompletionRequest(
                model=self._judge_model,
                messages=[
                    LLMMessage(
                        role="system",
                        content=(
                            "You are an evaluation judge for a memory benchmark. "
                            "Compare the prediction to the ground truth and decide if the "
                            "prediction contains the key facts. Be lenient about wording but "
                            "strict about factual correctness. Return JSON only with the shape "
                            '{"verdict": 1, "reasoning": "..."}.'
                        ),
                    ),
                    LLMMessage(
                        role="user",
                        content=(
                            f"Question: {question}\n"
                            f"Ground truth: {ground_truth}\n"
                            f"Prediction: {prediction}"
                        ),
                    ),
                ],
                temperature=0.0,
                max_output_tokens=256,
                metadata={
                    "purpose": "benchmark_judge",
                    "question": question,
                },
            )
        )
        return self._parse_result(response.output_text, response.model)

    def _parse_result(self, raw_output: str, response_model: str) -> ScoreResult:
        payload = self._extract_json_payload(raw_output)
        if payload is None:
            return ScoreResult(
                score=0,
                reasoning=_DEFAULT_REASONING,
                judge_model=response_model or self._judge_model,
            )

        verdict = payload.get("verdict")
        reasoning = str(payload.get("reasoning") or _DEFAULT_REASONING)
        score = 1 if verdict in (1, "1", True) else 0
        return ScoreResult(
            score=score,
            reasoning=reasoning,
            judge_model=response_model or self._judge_model,
        )

    @staticmethod
    def _extract_json_payload(raw_output: str) -> dict[str, object] | None:
        try:
            payload = json.loads(raw_output)
        except json.JSONDecodeError:
            match = _JSON_OBJECT_PATTERN.search(raw_output)
            if match is None:
                return None
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return payload if isinstance(payload, dict) else None
