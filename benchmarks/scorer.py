"""LLM-judge scoring for benchmark answers.

Three judge protocols live here (all benchmark-side; judges are exempt from the
engine prompt-fidelity rule by design):

- ``source_aware_strict`` — the historical behavior, kept unchanged as the
  answer-discipline signal. It sees the official evidence turns and rejects any
  unsupported addition, so it punishes true-but-out-of-scope extras.
- ``gold_only_lenient`` — Mem0-parity protocol (arXiv:2504.19413): question +
  ground truth + prediction only, lenient about wording and extra content.
  External comparability ONLY; it passes even false additions.
- ``memory_quality`` — the PRIMARY internal gate. Pass iff every requested fact
  is present and correct; extra content does NOT fail the answer when it is true
  in the conversation; fail on missing info, false additions, or misattributed
  references. Ground truth for extras is the conversation itself, so this judge
  is given the full source transcript in addition to the evidence turns.
"""

from __future__ import annotations

import enum
import json
import logging
import re
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

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


class JudgeProtocol(str, enum.Enum):
    """The three benchmark judge protocols, each with a distinct job."""

    SOURCE_AWARE_STRICT = "source_aware_strict"
    GOLD_ONLY_LENIENT = "gold_only_lenient"
    MEMORY_QUALITY = "memory_quality"


# Structured failure-reason vocabulary emitted by every protocol so later gates
# read structure instead of scanning free-text reasonings.
FAILURE_REASONS: tuple[str, ...] = (
    "unsupported_addition",
    "omission",
    "wrong_fact",
    "abstention",
    "temporal_precision",
    "misattribution",
    "other",
)


class JudgeVerdict(BaseModel):
    """Structured judge output for one predicted answer."""

    model_config = ConfigDict(extra="forbid")

    score: int = Field(ge=0, le=1)
    reasoning: str
    judge_model: str
    protocol: str
    failure_reason: str | None = None
    # Decomposition fields; populated by the memory_quality protocol only.
    missing_info: bool | None = None
    false_addition: bool | None = None
    misattribution: bool | None = None
    true_addition_only: bool | None = None

    def to_score_result(self) -> ScoreResult:
        """Downcast to the binary ScoreResult used by the live benchmark path."""
        return ScoreResult(
            score=self.score,
            reasoning=self.reasoning,
            judge_model=self.judge_model,
            protocol=self.protocol,
        )


class LLMJudgeScorer:
    """Score predictions using the configured Atagia LLM client and a protocol."""

    def __init__(
        self,
        llm_client: LLMClient[object],
        judge_model: str,
        protocol: JudgeProtocol = JudgeProtocol.SOURCE_AWARE_STRICT,
        max_output_tokens: int = GENERIC_JUDGE_MAX_OUTPUT_TOKENS,
    ) -> None:
        self._llm_client = llm_client
        self._judge_model = judge_model
        self._protocol = protocol
        # Reasoning-model judges spend thinking tokens inside this budget, so
        # high-effort specs can need more room than the generic default.
        self._max_output_tokens = max_output_tokens

    @property
    def judge_model(self) -> str:
        return self._judge_model

    @property
    def protocol(self) -> JudgeProtocol:
        return self._protocol

    async def score(
        self,
        question: str,
        prediction: str,
        ground_truth: str,
        source_evidence: Sequence[dict[str, Any]] | None = None,
        conversation_transcript: str | None = None,
    ) -> ScoreResult:
        """Return a binary judge verdict for one question-answer pair."""
        verdict = await self.evaluate(
            question=question,
            prediction=prediction,
            ground_truth=ground_truth,
            source_evidence=source_evidence,
            conversation_transcript=conversation_transcript,
        )
        return verdict.to_score_result()

    async def evaluate(
        self,
        *,
        question: str,
        prediction: str,
        ground_truth: str,
        source_evidence: Sequence[dict[str, Any]] | None = None,
        conversation_transcript: str | None = None,
    ) -> JudgeVerdict:
        """Return a structured judge verdict under the configured protocol."""
        instruction, user_content = self._build_prompt(
            question=question,
            prediction=prediction,
            ground_truth=ground_truth,
            source_evidence=source_evidence,
            conversation_transcript=conversation_transcript,
        )
        response = await self._llm_client.complete(
            LLMCompletionRequest(
                model=self._judge_model,
                messages=[
                    LLMMessage(role="system", content=instruction),
                    LLMMessage(role="user", content=user_content),
                ],
                max_output_tokens=self._max_output_tokens,
                metadata={
                    "purpose": "benchmark_judge",
                    "question": question,
                    "judge_protocol": self._protocol.value,
                    "source_evidence_used": bool(source_evidence),
                },
            )
        )
        return self._parse_result(response.output_text, response.model)

    def _build_prompt(
        self,
        *,
        question: str,
        prediction: str,
        ground_truth: str,
        source_evidence: Sequence[dict[str, Any]] | None,
        conversation_transcript: str | None,
    ) -> tuple[str, str]:
        if self._protocol is JudgeProtocol.GOLD_ONLY_LENIENT:
            return _gold_only_lenient_prompt(question, prediction, ground_truth)
        if self._protocol is JudgeProtocol.MEMORY_QUALITY:
            if conversation_transcript is None:
                raise ValueError(
                    "memory_quality protocol requires a conversation_transcript"
                )
            return _memory_quality_prompt(
                question,
                prediction,
                ground_truth,
                source_evidence,
                conversation_transcript,
            )
        return _source_aware_strict_prompt(
            question, prediction, ground_truth, source_evidence
        )

    def _parse_result(self, raw_output: str, response_model: str) -> JudgeVerdict:
        payload = self._extract_json_payload(raw_output)
        model_name = response_model or self._judge_model
        if payload is None:
            logger.warning(
                "Judge response could not be parsed: %s",
                raw_output[:500],
            )
            return JudgeVerdict(
                score=0,
                reasoning=_DEFAULT_REASONING,
                judge_model=model_name,
                protocol=self._protocol.value,
                failure_reason="other",
            )

        verdict = payload.get("verdict")
        reasoning = str(payload.get("reasoning") or _DEFAULT_REASONING)
        score = 1 if verdict in (1, "1", True) else 0
        failure_reason = _normalize_failure_reason(payload.get("failure_reason"), score)
        result = JudgeVerdict(
            score=score,
            reasoning=reasoning,
            judge_model=model_name,
            protocol=self._protocol.value,
            failure_reason=failure_reason,
        )
        if self._protocol is JudgeProtocol.MEMORY_QUALITY:
            result.missing_info = _coerce_optional_bool(payload.get("missing_info"))
            result.false_addition = _coerce_optional_bool(payload.get("false_addition"))
            result.misattribution = _coerce_optional_bool(
                payload.get("misattribution")
            )
            result.true_addition_only = _coerce_optional_bool(
                payload.get("true_addition_only")
            )
        return result

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


def _normalize_failure_reason(raw_value: object, score: int) -> str | None:
    if score == 1:
        return None
    text = str(raw_value or "").strip().lower()
    if text in FAILURE_REASONS:
        return text
    return "other"


def _coerce_optional_bool(raw_value: object) -> bool | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        return raw_value
    text = str(raw_value).strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no"):
        return False
    return None


def _failure_reason_clause(allowed: Sequence[str]) -> str:
    return (
        'When the verdict is 0, also set "failure_reason" to one of: '
        + ", ".join(allowed)
        + ". Omit it (or use null) when the verdict is 1."
    )


def _source_aware_strict_prompt(
    question: str,
    prediction: str,
    ground_truth: str,
    source_evidence: Sequence[dict[str, Any]] | None,
) -> tuple[str, str]:
    evidence_block = _format_source_evidence(source_evidence)
    failure_clause = _failure_reason_clause(FAILURE_REASONS)
    if evidence_block:
        instruction = (
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
            '{"verdict": 1, "reasoning": "...", "failure_reason": null}. '
            + failure_clause
        )
        user_content = (
            f"Question: {question}\n"
            f"Ground truth: {ground_truth}\n"
            f"Official source evidence:\n{evidence_block}\n\n"
            f"Prediction: {prediction}"
        )
    else:
        instruction = (
            "You are an evaluation judge for a memory benchmark. "
            "Compare the prediction to the ground truth and decide if the "
            "prediction contains the key facts. Be lenient about wording but "
            "strict about factual correctness. Return JSON only with the shape "
            '{"verdict": 1, "reasoning": "...", "failure_reason": null}. '
            + failure_clause
        )
        user_content = (
            f"Question: {question}\n"
            f"Ground truth: {ground_truth}\n"
            f"Prediction: {prediction}"
        )
    return instruction, user_content


def _gold_only_lenient_prompt(
    question: str,
    prediction: str,
    ground_truth: str,
) -> tuple[str, str]:
    allowed = ("omission", "wrong_fact", "abstention", "temporal_precision", "other")
    instruction = (
        "You are an evaluation judge for a memory benchmark, applying a lenient "
        "gold-only protocol used for external comparability with published "
        "results. You are given only the question, the ground-truth answer, and "
        "the model's prediction; you are NOT given the source conversation. "
        "Decide whether the prediction conveys the key facts of the ground "
        "truth. Be lenient about wording, phrasing, and format. A prediction "
        "PASSES (verdict 1) as long as it contains the information in the "
        "ground truth and does not contradict it. Do not penalize additional "
        "detail or extra content: only the presence and correctness of the "
        "ground-truth information matters. Return JSON only with the shape "
        '{"verdict": 1, "reasoning": "...", "failure_reason": null}. '
        + _failure_reason_clause(allowed)
    )
    user_content = (
        f"Question: {question}\n"
        f"Ground truth: {ground_truth}\n"
        f"Prediction: {prediction}"
    )
    return instruction, user_content


def _memory_quality_prompt(
    question: str,
    prediction: str,
    ground_truth: str,
    source_evidence: Sequence[dict[str, Any]] | None,
    conversation_transcript: str,
) -> tuple[str, str]:
    allowed = (
        "omission",
        "wrong_fact",
        "unsupported_addition",
        "misattribution",
        "temporal_precision",
        "abstention",
        "other",
    )
    instruction = (
        "You are an evaluation judge measuring MEMORY QUALITY for a memory "
        "engine. You are given the question, the compact ground-truth answer, "
        "the model's prediction, the official source-evidence turns, and the "
        "FULL source conversation. Treat the full conversation as the ground "
        "truth for anything not stated in the compact answer. Judge the content "
        "and memory correctness only, not writing style or verbosity. A "
        "prediction PASSES (verdict 1) if and only if every piece of "
        "information the question asks for is present and correct. Apply these "
        "rules: (a) Extra content beyond the ground truth does NOT fail the "
        "answer as long as that extra content is TRUE according to the "
        "conversation. (b) FAIL if any requested information is missing. "
        "(c) FAIL if the prediction states a fact that is false or unsupported "
        "by the conversation (a false addition). (d) FAIL if the prediction "
        "attributes a fact to the wrong person or entity (misattribution or a "
        "crossed reference), even when that fact is true for someone else in "
        "the conversation. Use the full conversation to verify whether "
        "additions are true and correctly attributed. Return JSON only with the "
        "shape "
        '{"verdict": 1, "missing_info": false, "false_addition": false, '
        '"misattribution": false, "true_addition_only": false, '
        '"failure_reason": null, "reasoning": "..."}. Field meanings: '
        "missing_info = some requested information is absent; "
        "false_addition = the prediction adds a fact that is not true in the "
        "conversation; misattribution = the prediction assigns a fact to the "
        "wrong person or entity; true_addition_only = the prediction contains "
        "all requested information correctly AND its only deviation from the "
        "compact ground truth is extra content that IS true in the conversation "
        "(it would fail a strict source-slice judge but is memory-correct). "
        + _failure_reason_clause(allowed)
    )
    evidence_block = _format_source_evidence(source_evidence) or "(none provided)"
    # The transcript is placed first so that, across all questions of one
    # conversation, the system instruction + transcript form an identical prefix
    # that provider prefix-caching can reuse (the per-question fields follow).
    user_content = (
        f"Full source conversation:\n{conversation_transcript}\n\n"
        f"--- Answer to evaluate ---\n"
        f"Question: {question}\n"
        f"Ground truth: {ground_truth}\n"
        f"Official source evidence turns:\n{evidence_block}\n\n"
        f"Prediction: {prediction}"
    )
    return instruction, user_content


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
        session_id = str(raw_item.get("session_id") or "").strip()
        text = str(raw_item.get("text") or "").strip()
        blip_caption = str(raw_item.get("blip_caption") or "").strip()
        attachment_text = str(raw_item.get("attachment_text") or "").strip()
        body_parts: list[str] = []
        if text:
            body_parts.append(text)
        if blip_caption:
            body_parts.append(f"[Image caption]\n{blip_caption}")
        if attachment_text and attachment_text != blip_caption:
            body_parts.append(f"[Attachment text]\n{attachment_text}")
        if not body_parts:
            continue
        header_parts = [f"#{index}"]
        if turn_id:
            header_parts.append(f"turn_id={turn_id}")
        if session_id:
            header_parts.append(f"session_id={session_id}")
        if timestamp:
            header_parts.append(f"timestamp={timestamp}")
        if speaker:
            header_parts.append(f"speaker={speaker}")
        lines.append(f"{' | '.join(header_parts)}\n" + "\n\n".join(body_parts))
    return "\n\n".join(lines)
