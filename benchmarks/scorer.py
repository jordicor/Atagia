"""LLM-judge scoring for benchmark answers."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from typing import Any

from benchmarks.base import ScoreResult
from atagia.core.llm_output_limits import GENERIC_JUDGE_MAX_OUTPUT_TOKENS
from atagia.services.llm_client import LLMClient, LLMCompletionRequest, LLMMessage

try:
    from ai_json_cleanroom import validate_ai_json
except ImportError:  # pragma: no cover - exercised in isolated benchmark envs.
    validate_ai_json = None

_DEFAULT_REASONING = "Judge response could not be parsed as valid verdict JSON."
_JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
logger = logging.getLogger(__name__)


class LLMJudgeScorer:
    """Score predictions using the configured Atagia LLM client."""

    def __init__(self, llm_client: LLMClient[object], judge_model: str) -> None:
        self._llm_client = llm_client
        self._judge_model = judge_model

    @property
    def judge_model(self) -> str:
        return self._judge_model

    async def score(
        self,
        question: str,
        prediction: str,
        ground_truth: str,
        source_evidence: Sequence[dict[str, Any]] | None = None,
    ) -> ScoreResult:
        """Return a binary judge verdict for one question-answer pair."""
        evidence_block = _format_source_evidence(source_evidence)
        if evidence_block:
            judging_instruction = (
                "You are an evaluation judge for a memory benchmark. "
                "Compare the prediction to the ground truth and official source "
                "evidence, then decide if the prediction contains the key facts. "
                "The ground truth may be a compact expected answer rather than a "
                "complete paraphrase of every source-supported detail. Be lenient "
                "about wording. Accept additional detail only when it is directly "
                "supported by the official source evidence or by necessary temporal "
                "inference from the source timestamps. Temporal specificity rule: "
                "if the ground truth omits a year or another date component, do "
                "not reject a prediction merely for adding that component when the "
                "official source timestamp and wording support the natural "
                "inference. For future-oriented events stated relative to a source "
                "timestamp, prefer the next applicable future occurrence unless "
                "the source evidence says otherwise; do not reinterpret a "
                "future-oriented event as already past only because the compact "
                "ground truth omits the year. Reject unsupported or contradictory "
                "added facts. Return JSON only with the shape "
                '{"verdict": 1, "reasoning": "..."}.'
            )
            user_content = (
                f"Question: {question}\n"
                f"Ground truth: {ground_truth}\n"
                f"Official source evidence:\n{evidence_block}\n\n"
                f"Prediction: {prediction}"
            )
        else:
            judging_instruction = (
                "You are an evaluation judge for a memory benchmark. "
                "Compare the prediction to the ground truth and decide if the "
                "prediction contains the key facts. Be lenient about wording but "
                "strict about factual correctness. Return JSON only with the shape "
                '{"verdict": 1, "reasoning": "..."}.'
            )
            user_content = (
                f"Question: {question}\n"
                f"Ground truth: {ground_truth}\n"
                f"Prediction: {prediction}"
            )
        response = await self._llm_client.complete(
            LLMCompletionRequest(
                model=self._judge_model,
                messages=[
                    LLMMessage(
                        role="system",
                        content=judging_instruction,
                    ),
                    LLMMessage(
                        role="user",
                        content=user_content,
                    ),
                ],
                max_output_tokens=GENERIC_JUDGE_MAX_OUTPUT_TOKENS,
                metadata={
                    "purpose": "benchmark_judge",
                    "question": question,
                    "source_evidence_used": bool(evidence_block),
                },
            )
        )
        return self._parse_result(response.output_text, response.model)

    def _parse_result(self, raw_output: str, response_model: str) -> ScoreResult:
        payload = self._extract_json_payload(raw_output)
        if payload is None:
            logger.warning(
                "Judge response could not be parsed: %s",
                raw_output[:500],
            )
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
        if validate_ai_json is not None:
            result = validate_ai_json(raw_output)
            if result.json_valid and isinstance(result.data, dict):
                return result.data
            return None
        payload = _fallback_json_payload(raw_output)
        if isinstance(payload, dict):
            return payload
        return None


def _fallback_json_payload(raw_output: str) -> object | None:
    """Parse simple JSON judge payloads without optional cleanroom dependency."""
    candidates = [raw_output.strip()]
    candidates.extend(match.group(1).strip() for match in _JSON_FENCE_PATTERN.finditer(raw_output))
    start = raw_output.find("{")
    end = raw_output.rfind("}")
    if start >= 0 and end > start:
        candidates.append(raw_output[start : end + 1])
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _format_source_evidence(source_evidence: Sequence[dict[str, Any]] | None) -> str:
    """Render official benchmark evidence for the judge prompt."""
    if not source_evidence:
        return ""
    lines: list[str] = []
    for index, raw_item in enumerate(source_evidence, start=1):
        if not isinstance(raw_item, dict):
            continue
        turn_id = str(raw_item.get("turn_id") or "").strip()
        timestamp = str(raw_item.get("timestamp") or "").strip()
        speaker = str(raw_item.get("speaker") or raw_item.get("role") or "").strip()
        text = str(raw_item.get("text") or "").strip()
        if not text:
            continue
        header_parts = [f"#{index}"]
        if turn_id:
            header_parts.append(f"turn_id={turn_id}")
        if timestamp:
            header_parts.append(f"timestamp={timestamp}")
        if speaker:
            header_parts.append(f"speaker={speaker}")
        lines.append(f"{' | '.join(header_parts)}\n{text}")
    return "\n\n".join(lines)
