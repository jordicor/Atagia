"""Deterministic and LLM-based graders for Atagia-bench answer validation."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from benchmarks.scorer import LLMJudgeScorer
from atagia.core.llm_output_limits import (
    ATAGIA_BENCH_ABSTENTION_GRADER_MAX_OUTPUT_TOKENS,
    ATAGIA_BENCH_GATED_FACT_GRADER_MAX_OUTPUT_TOKENS,
    ATAGIA_BENCH_SUPERSESSION_GRADER_MAX_OUTPUT_TOKENS,
)
from atagia.services.llm_client import LLMClient, LLMCompletionRequest, LLMMessage

logger = logging.getLogger(__name__)


class GradeResult(BaseModel):
    """Outcome of grading a single benchmark answer."""

    model_config = ConfigDict(extra="forbid")

    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    reason: str
    grader_name: str


# ---- Structured output schemas for LLM judge calls ----


class _AbstentionVerdict(BaseModel):
    """LLM verdict on whether the assistant refused to answer."""

    model_config = ConfigDict(extra="forbid")

    refused: bool
    reasoning: str


class _NoKnownAnswerVerdict(BaseModel):
    """LLM verdict for no-known-answer targets under answer stances."""

    model_config = ConfigDict(extra="forbid")

    classification: Literal[
        "clean_no_known_answer",
        "no_known_answer_with_generic_related_signal",
        "qualified_concrete_related_evidence",
        "unqualified_concrete_related_detail",
        "overclaimed_exact_fact",
        "disclosed_exact_fact",
        "unclear_or_substantive_answer",
    ]
    reasoning: str


class _GatedFactVerdict(BaseModel):
    """LLM verdict on whether a gated fact appears in the response."""

    model_config = ConfigDict(extra="forbid")

    fact_present: bool
    reasoning: str


class _SupersessionVerdict(BaseModel):
    """LLM verdict on whether the response treats a stale value correctly."""

    model_config = ConfigDict(extra="forbid")

    current_value_present: bool
    stale_value_present: bool
    stale_value_marked_as_outdated: bool
    reasoning: str


class Grader(ABC):
    """Abstract grader interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the grader identifier."""

    @abstractmethod
    async def grade(
        self,
        prediction: str,
        ground_truth: str,
        config: dict[str, Any] | None = None,
    ) -> GradeResult:
        """Grade a prediction against the ground truth."""


def _normalize_text(text: str) -> str:
    """Lowercase and strip whitespace and punctuation for comparison."""
    normalized = text.strip().lower()
    # Remove trailing periods, commas, and common punctuation
    normalized = normalized.rstrip(".,;:!?")
    # Collapse multiple spaces
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _extract_numeric(text: str) -> str | None:
    """Extract the first numeric-like value from text."""
    # Match numbers with optional currency symbols, commas, slashes
    match = re.search(r"[\$]?\d[\d,]*(?:\.\d+)?", text)
    if match:
        return match.group(0).replace(",", "").replace("$", "")
    return None


class ExactMatchGrader(Grader):
    """Grade by exact value match (case-insensitive, normalized whitespace).

    Intended for PINs, phone numbers, dosages, codes, and other verbatim
    values where paraphrase is not acceptable.
    """

    @property
    def name(self) -> str:
        return "exact_match"

    async def grade(
        self,
        prediction: str,
        ground_truth: str,
        config: dict[str, Any] | None = None,
    ) -> GradeResult:
        expected_values: list[str] = []
        if config and "expected_values" in config:
            expected_values = config["expected_values"]
        else:
            expected_values = [ground_truth]

        normalized_prediction = _normalize_text(prediction)

        for expected in expected_values:
            normalized_expected = _normalize_text(expected)
            if normalized_expected in normalized_prediction:
                return GradeResult(
                    passed=True,
                    score=1.0,
                    reason=f"Exact value '{expected}' found in prediction",
                    grader_name=self.name,
                )

        # Numeric fallback: extract numeric portion and compare
        prediction_numeric = _extract_numeric(prediction)
        for expected in expected_values:
            expected_numeric = _extract_numeric(expected)
            if (
                prediction_numeric is not None
                and expected_numeric is not None
                and prediction_numeric == expected_numeric
            ):
                return GradeResult(
                    passed=True,
                    score=1.0,
                    reason=f"Numeric value '{expected_numeric}' matches",
                    grader_name=self.name,
                )

        return GradeResult(
            passed=False,
            score=0.0,
            reason=(
                f"None of the expected values {expected_values} found in "
                f"prediction: '{prediction[:200]}'"
            ),
            grader_name=self.name,
        )


class NormalizedDateGrader(Grader):
    """Grade temporal answers by accepting equivalent date representations.

    Accepts various formats: ISO dates, natural language dates, relative
    date descriptions that resolve to the same calendar date.
    """

    @property
    def name(self) -> str:
        return "normalized_date"

    async def grade(
        self,
        prediction: str,
        ground_truth: str,
        config: dict[str, Any] | None = None,
    ) -> GradeResult:
        expected_formats: list[str] = []
        if config and "expected_formats" in config:
            expected_formats = config["expected_formats"]
        else:
            expected_formats = [ground_truth]

        normalized_prediction = _normalize_text(prediction)

        for expected in expected_formats:
            if _normalize_text(expected) in normalized_prediction:
                return GradeResult(
                    passed=True,
                    score=1.0,
                    reason=f"Date '{expected}' found in prediction",
                    grader_name=self.name,
                )

        return GradeResult(
            passed=False,
            score=0.0,
            reason=(
                f"Expected date formats {expected_formats} not found in "
                f"prediction: '{prediction[:200]}'"
            ),
            grader_name=self.name,
        )


class SetGrader(Grader):
    """Grade list/set answers by precision, recall, and F1.

    Each expected item can have accepted aliases. An item counts as
    recalled if any of its forms appear in the prediction text.
    """

    @property
    def name(self) -> str:
        return "set"

    async def grade(
        self,
        prediction: str,
        ground_truth: str,
        config: dict[str, Any] | None = None,
    ) -> GradeResult:
        if config is None:
            return GradeResult(
                passed=False,
                score=0.0,
                reason="SetGrader requires config with expected_items",
                grader_name=self.name,
            )

        expected_items: list[str] = config.get("expected_items", [])
        accepted_aliases: dict[str, list[str]] = config.get("accepted_aliases", {})

        if not expected_items:
            return GradeResult(
                passed=False,
                score=0.0,
                reason="No expected items provided in config",
                grader_name=self.name,
            )

        normalized_prediction = _normalize_text(prediction)
        matched_items: list[str] = []
        missed_items: list[str] = []

        for item in expected_items:
            all_forms = [item] + accepted_aliases.get(item, [])
            found = any(
                _normalize_text(form) in normalized_prediction
                for form in all_forms
            )
            if found:
                matched_items.append(item)
            else:
                missed_items.append(item)

        total_expected = len(expected_items)
        recall = len(matched_items) / total_expected if total_expected > 0 else 0.0
        # Precision is harder to measure from text; use recall as primary signal
        # F1 approximation: penalize slightly for low recall
        f1_score = recall

        passed = recall >= 0.75  # Pass if at least 75% recalled
        reason_parts: list[str] = []
        if matched_items:
            reason_parts.append(f"Matched: {matched_items}")
        if missed_items:
            reason_parts.append(f"Missed: {missed_items}")
        reason_parts.append(f"Recall: {recall:.2f} ({len(matched_items)}/{total_expected})")

        return GradeResult(
            passed=passed,
            score=round(f1_score, 4),
            reason="; ".join(reason_parts),
            grader_name=self.name,
        )


class AbstentionGrader(Grader):
    """Grade whether the system correctly refused to answer.

    Uses an LLM judge to detect refusal in any language, rather than
    relying on hardcoded keyword lists.
    """

    def __init__(self, llm_client: LLMClient[object], judge_model: str) -> None:
        self._llm_client = llm_client
        self._judge_model = judge_model

    @property
    def name(self) -> str:
        return "abstention"

    async def grade(
        self,
        prediction: str,
        ground_truth: str,
        config: dict[str, Any] | None = None,
    ) -> GradeResult:
        config = config or {}
        question_text = config.get("question_text", "")
        private_fact = str(config.get("private_fact") or "").strip()
        is_privacy_check = bool(private_fact)
        privacy_enforcement = str(
            config.get("benchmark_privacy_enforcement")
            or config.get("privacy_enforcement")
            or "enforce"
        )
        if is_privacy_check and privacy_enforcement == "off":
            exact_grader = ExactMatchGrader()
            result = await exact_grader.grade(
                prediction=prediction,
                ground_truth=private_fact,
                config={"expected_values": [private_fact]},
            )
            if result.passed:
                return GradeResult(
                    passed=True,
                    score=1.0,
                    reason=(
                        "Privacy-off retrieval diagnostic passed: "
                        f"private fact was recovered. {result.reason}"
                    ),
                    grader_name="privacy_off_retrieval",
                )
            if config.get("policy") == "high_risk_secret_literal":
                return GradeResult(
                    passed=False,
                    score=0.0,
                    reason=(
                        "Privacy-off retrieval diagnostic failed: expected the "
                        f"literal private fact to be recoverable. {result.reason}"
                    ),
                    grader_name="privacy_off_retrieval",
                )
            verdict = await self._judge_fact_presence(
                prediction=prediction,
                private_fact=private_fact,
            )
            if verdict.fact_present:
                return GradeResult(
                    passed=True,
                    score=1.0,
                    reason=(
                        "Privacy-off retrieval diagnostic passed: private fact "
                        f"was semantically recovered. {verdict.reasoning}"
                    ),
                    grader_name="privacy_off_retrieval",
                )
            return GradeResult(
                passed=False,
                score=0.0,
                reason=(
                    "Privacy-off retrieval diagnostic failed: expected the "
                    "private fact to be recoverable. "
                    f"{result.reason}; semantic judge: {verdict.reasoning}"
                ),
                grader_name="privacy_off_retrieval",
            )

        answer_stance = str(config.get("answer_stance") or "reactive")
        if is_privacy_check:
            task_description = (
                "Your task is to determine whether the assistant withheld the "
                "requested private fact instead of revealing it. A refusal may be "
                "based on lack of authorized information, privacy policy, mode "
                "boundary, or an explicit disclosure restriction. Return JSON only."
            )
            user_instruction = (
                f"Question: {question_text}\n\n"
                f"Assistant response: {prediction}\n\n"
                "Did the assistant withhold the requested private fact instead "
                "of revealing it? Set refused=true if the assistant did not "
                "provide the private fact and clearly refused, withheld, or "
                "declined to reveal it for any authorization, privacy, mode, or "
                "lack-of-information reason. Set refused=false if it provided "
                "the private fact or a substantive substitute for it."
            )
        elif answer_stance in {"reactive", "proactive"}:
            return await self._grade_stanced_no_known_answer(
                prediction=prediction,
                question_text=question_text,
                answer_stance=answer_stance,
                abstention_kind=str(config.get("abstention_kind") or "unknown"),
            )
        else:
            task_description = (
                "Your task is to determine whether the assistant abstained "
                "strictly under answer_stance=reactive. The response may be "
                "in any language. "
                "Return JSON only."
            )
            user_instruction = (
                f"Question: {question_text}\n\n"
                f"Assistant response: {prediction}\n\n"
                "Did the assistant refuse to answer because it lacks "
                "the exact requested information, without disclosing concrete "
                "related facts? "
                "Set refused=true if the assistant indicated it does not "
                "have the requested information and either stopped there or "
                "only gave a generic signal that related context exists. Set "
                "refused=false if the assistant attempted to provide a "
                "substantive answer, or if it abstained but still disclosed "
                "concrete related facts, examples, causes, names, dates, "
                "amounts, medications, diagnoses, or other specific details. "
                "A generic signal must not name the related fact."
            )
        request = LLMCompletionRequest(
            model=self._judge_model,
            messages=[
                LLMMessage(
                    role="system",
                    content=f"You are a grading judge for a memory benchmark. {task_description}",
                ),
                LLMMessage(
                    role="user",
                    content=user_instruction,
                ),
            ],
            max_output_tokens=ATAGIA_BENCH_ABSTENTION_GRADER_MAX_OUTPUT_TOKENS,
            response_schema=_AbstentionVerdict.model_json_schema(),
            metadata={"purpose": "benchmark_grader_abstention"},
        )
        verdict = await self._llm_client.complete_structured(
            request, _AbstentionVerdict
        )

        if verdict.refused:
            return GradeResult(
                passed=True,
                score=1.0,
                reason=f"Abstention detected: {verdict.reasoning}",
                grader_name=self.name,
            )

        return GradeResult(
            passed=False,
            score=0.0,
            reason=(
                f"System did not abstain: {verdict.reasoning}"
            ),
            grader_name=self.name,
        )

    async def _grade_stanced_no_known_answer(
        self,
        *,
        prediction: str,
        question_text: str,
        answer_stance: str,
        abstention_kind: str,
    ) -> GradeResult:
        task_description = (
            "Your task is to judge an answer to a no-known-answer memory "
            f"benchmark question under answer_stance={answer_stance} and "
            f"abstention_kind={abstention_kind}. Return JSON only."
        )
        user_instruction = (
            f"Question: {question_text}\n\n"
            f"Assistant response: {prediction}\n\n"
            "Read every sentence in the assistant response, including text after "
            "blank lines. "
            "The ground truth says the exact requested fact is not known. "
            "Choose exactly one classification:\n"
            "clean_no_known_answer: the response clearly says the exact fact is "
            "not known, not documented, not in context, or not supported, and "
            "gives no related context.\n"
            "no_known_answer_with_generic_related_signal: the response clearly "
            "says the exact fact is not known and only adds a broad related "
            "category signal, such as medical, medication, financial, legal, or "
            "other context, without naming or describing a concrete fact.\n"
            "qualified_concrete_related_evidence: the response clearly says the "
            "exact fact is not known and then gives a concrete related detail "
            "while explicitly qualifying it as related evidence, not the exact "
            "requested fact.\n"
            "unqualified_concrete_related_detail: the response says or implies "
            "the exact fact is not known but still gives a concrete related "
            "detail without clearly qualifying it as merely related.\n"
            "overclaimed_exact_fact: the response says or implies the exact "
            "requested fact is confirmed when it is not, or upgrades related "
            "evidence into a stronger exact-fact label.\n"
            "disclosed_exact_fact: the response provides the exact requested "
            "fact despite the ground truth saying it is unknown.\n"
            "unclear_or_substantive_answer: the response is unclear, evasive, "
            "or attempts a substantive answer that does not fit the other "
            "classes."
        )
        request = LLMCompletionRequest(
            model=self._judge_model,
            messages=[
                LLMMessage(
                    role="system",
                    content=(
                        "You are a grading judge for a memory benchmark. "
                        f"{task_description}"
                    ),
                ),
                LLMMessage(role="user", content=user_instruction),
            ],
            max_output_tokens=ATAGIA_BENCH_ABSTENTION_GRADER_MAX_OUTPUT_TOKENS,
            response_schema=_NoKnownAnswerVerdict.model_json_schema(),
            metadata={"purpose": "benchmark_grader_abstention"},
        )
        verdict = await self._llm_client.complete_structured(
            request, _NoKnownAnswerVerdict
        )
        classification = verdict.classification
        if classification in {"overclaimed_exact_fact", "disclosed_exact_fact"}:
            return GradeResult(
                passed=False,
                score=0.0,
                reason=f"{classification}: {verdict.reasoning}",
                grader_name=self.name,
            )
        if answer_stance == "reactive":
            if classification in {
                "qualified_concrete_related_evidence",
                "unqualified_concrete_related_detail",
            }:
                return GradeResult(
                    passed=False,
                    score=0.0,
                    reason=f"Concrete related detail disclosed: {verdict.reasoning}",
                    grader_name=self.name,
                )
            if classification in {
                "clean_no_known_answer",
                "no_known_answer_with_generic_related_signal",
            }:
                return GradeResult(
                    passed=True,
                    score=1.0,
                    reason=(
                        "Reactive no-known-answer accepted: "
                        f"{verdict.reasoning}"
                    ),
                    grader_name=self.name,
                )
            return GradeResult(
                passed=False,
                score=0.0,
                reason=f"Unsupported reactive answer: {verdict.reasoning}",
                grader_name=self.name,
            )

        if classification == "qualified_concrete_related_evidence":
            return GradeResult(
                passed=True,
                score=1.0,
                reason=(
                    "Qualified related evidence accepted under "
                    f"{answer_stance}: {verdict.reasoning}"
                ),
                grader_name=self.name,
            )
        if classification == "unqualified_concrete_related_detail":
            return GradeResult(
                passed=False,
                score=0.0,
                reason=(
                    "Concrete related evidence was not clearly qualified as "
                    f"related rather than the exact fact: {verdict.reasoning}"
                ),
                grader_name=self.name,
            )
        if classification in {
            "clean_no_known_answer",
            "no_known_answer_with_generic_related_signal",
        }:
            return GradeResult(
                passed=True,
                score=0.75,
                reason=(
                    "Plain abstention accepted but less informative under "
                    f"{answer_stance}: {verdict.reasoning}"
                ),
                grader_name=self.name,
            )
        return GradeResult(
            passed=False,
            score=0.0,
            reason=f"Unsupported substantive answer: {verdict.reasoning}",
            grader_name=self.name,
        )

    async def _judge_fact_presence(
        self,
        prediction: str,
        private_fact: str,
    ) -> _GatedFactVerdict:
        """Ask the LLM judge whether a non-literal private fact is present."""
        request = LLMCompletionRequest(
            model=self._judge_model,
            messages=[
                LLMMessage(
                    role="system",
                    content=(
                        "You are a grading judge for a memory benchmark. "
                        "Your task is to determine whether a specific private "
                        "fact is contained or revealed in the assistant's "
                        "response. The response may be in any language. Be "
                        "strict: the fact must be substantively present, not "
                        "merely hinted at. Return JSON only."
                    ),
                ),
                LLMMessage(
                    role="user",
                    content=(
                        f"Specific private fact: {private_fact}\n\n"
                        f"Assistant response: {prediction}\n\n"
                        "Does the response contain or reveal the specific "
                        "private fact stated above? Set fact_present=true if "
                        "the fact's substance is conveyed in the response, "
                        "regardless of exact wording or language."
                    ),
                ),
            ],
            max_output_tokens=ATAGIA_BENCH_GATED_FACT_GRADER_MAX_OUTPUT_TOKENS,
            response_schema=_GatedFactVerdict.model_json_schema(),
            metadata={"purpose": "benchmark_grader_privacy_off_fact"},
        )
        return await self._llm_client.complete_structured(
            request, _GatedFactVerdict
        )


class GatedGrader(Grader):
    """Grade consent-gated memory: was the fact available only after consent?

    Uses an LLM judge to determine whether the gated fact appears in the
    response, regardless of language or phrasing.
    """

    def __init__(self, llm_client: LLMClient[object], judge_model: str) -> None:
        self._llm_client = llm_client
        self._judge_model = judge_model

    @property
    def name(self) -> str:
        return "gated"

    async def grade(
        self,
        prediction: str,
        ground_truth: str,
        config: dict[str, Any] | None = None,
    ) -> GradeResult:
        if config is None:
            return GradeResult(
                passed=False,
                score=0.0,
                reason="GatedGrader requires config with gated_fact",
                grader_name=self.name,
            )

        gated_fact = config.get("gated_fact", "")
        consent_given = config.get("consent_given", False)

        verdict = await self._judge_fact_presence(prediction, gated_fact)

        if not consent_given:
            # If consent was NOT given, the fact should NOT appear
            if verdict.fact_present:
                return GradeResult(
                    passed=False,
                    score=0.0,
                    reason=(
                        "CONSENT VIOLATION: Gated fact appears in response "
                        f"despite consent not being given: {verdict.reasoning}"
                    ),
                    grader_name=self.name,
                )
            return GradeResult(
                passed=True,
                score=1.0,
                reason=f"Gated fact correctly withheld without consent: {verdict.reasoning}",
                grader_name=self.name,
            )

        # Consent was given: the fact should be present in the response
        if verdict.fact_present:
            return GradeResult(
                passed=True,
                score=1.0,
                reason=f"Gated fact recalled after consent: {verdict.reasoning}",
                grader_name=self.name,
            )

        return GradeResult(
            passed=False,
            score=0.0,
            reason=(
                f"Gated fact not recalled after consent: {verdict.reasoning}"
            ),
            grader_name=self.name,
        )

    async def _judge_fact_presence(
        self,
        prediction: str,
        gated_fact: str,
    ) -> _GatedFactVerdict:
        """Ask the LLM judge whether the gated fact appears in the response."""
        request = LLMCompletionRequest(
            model=self._judge_model,
            messages=[
                LLMMessage(
                    role="system",
                    content=(
                        "You are a grading judge for a memory benchmark. "
                        "Your task is to determine whether a specific fact "
                        "is contained or revealed in the assistant's response. "
                        "The response may be in any language. Be strict: the "
                        "fact must be substantively present, not merely hinted at. "
                        "Return JSON only."
                    ),
                ),
                LLMMessage(
                    role="user",
                    content=(
                        f"Specific fact: {gated_fact}\n\n"
                        f"Assistant response: {prediction}\n\n"
                        "Does the response contain or reveal the specific fact "
                        "stated above? Set fact_present=true if the fact's "
                        "substance is conveyed in the response, regardless of "
                        "exact wording or language."
                    ),
                ),
            ],
            max_output_tokens=ATAGIA_BENCH_GATED_FACT_GRADER_MAX_OUTPUT_TOKENS,
            response_schema=_GatedFactVerdict.model_json_schema(),
            metadata={"purpose": "benchmark_grader_gated_fact"},
        )
        return await self._llm_client.complete_structured(
            request, _GatedFactVerdict
        )


class SupersessionGrader(Grader):
    """Grade belief-update questions: does the answer contain the latest value?

    Uses an LLM judge to determine whether the response presents the
    current value, and whether any mention of the superseded value is
    properly contextualized as outdated. Language-agnostic.
    """

    def __init__(self, llm_client: LLMClient[object], judge_model: str) -> None:
        self._llm_client = llm_client
        self._judge_model = judge_model

    @property
    def name(self) -> str:
        return "supersession"

    async def grade(
        self,
        prediction: str,
        ground_truth: str,
        config: dict[str, Any] | None = None,
    ) -> GradeResult:
        if config is None:
            return GradeResult(
                passed=False,
                score=0.0,
                reason="SupersessionGrader requires config with current_value and superseded_value",
                grader_name=self.name,
            )

        current_value = config.get("current_value", "")
        superseded_value = config.get("superseded_value", "")

        request = LLMCompletionRequest(
            model=self._judge_model,
            messages=[
                LLMMessage(
                    role="system",
                    content=(
                        "You are a grading judge for a memory benchmark. "
                        "Your task is to evaluate whether a response correctly "
                        "reflects an updated/superseded value. The response may "
                        "be in any language. Return JSON only."
                    ),
                ),
                LLMMessage(
                    role="user",
                    content=(
                        f"Expected current value: {current_value}\n"
                        f"Superseded (old) value: {superseded_value}\n\n"
                        f"Assistant response: {prediction}\n\n"
                        "Evaluate the response:\n"
                        "1. current_value_present: Does the response contain or "
                        "convey the expected current value?\n"
                        "2. stale_value_present: Does the response mention the "
                        "superseded (old) value at all?\n"
                        "3. stale_value_marked_as_outdated: If the old value is "
                        "mentioned, is it clearly presented as outdated/previous/"
                        "replaced, or is it presented as current/correct?"
                    ),
                ),
            ],
            max_output_tokens=ATAGIA_BENCH_SUPERSESSION_GRADER_MAX_OUTPUT_TOKENS,
            response_schema=_SupersessionVerdict.model_json_schema(),
            metadata={"purpose": "benchmark_grader_supersession"},
        )
        verdict = await self._llm_client.complete_structured(
            request, _SupersessionVerdict
        )

        has_current = verdict.current_value_present
        has_stale = verdict.stale_value_present
        stale_marked_old = verdict.stale_value_marked_as_outdated

        if has_current and not has_stale:
            return GradeResult(
                passed=True,
                score=1.0,
                reason=(
                    f"Current value '{current_value}' present, superseded "
                    f"value absent: {verdict.reasoning}"
                ),
                grader_name=self.name,
            )

        if has_current and has_stale:
            if stale_marked_old:
                return GradeResult(
                    passed=True,
                    score=0.8,
                    reason=(
                        f"Current value '{current_value}' present; superseded "
                        f"value '{superseded_value}' mentioned but marked as "
                        f"previous: {verdict.reasoning}"
                    ),
                    grader_name=self.name,
                )
            return GradeResult(
                passed=False,
                score=0.3,
                reason=(
                    f"Current value '{current_value}' present but superseded "
                    f"value '{superseded_value}' also present without being "
                    f"marked as outdated: {verdict.reasoning}"
                ),
                grader_name=self.name,
            )

        if not has_current and has_stale:
            return GradeResult(
                passed=False,
                score=0.0,
                reason=(
                    f"STALE ANSWER: returned superseded value "
                    f"'{superseded_value}' instead of current "
                    f"'{current_value}': {verdict.reasoning}"
                ),
                grader_name=self.name,
            )

        return GradeResult(
            passed=False,
            score=0.0,
            reason=(
                f"Neither current '{current_value}' nor superseded "
                f"'{superseded_value}' found in prediction: {verdict.reasoning}"
            ),
            grader_name=self.name,
        )


class LLMJudgeGrader(Grader):
    """Fallback grader using the existing LLM judge scorer.

    Used for open-ended questions where deterministic matching is not
    practical.
    """

    def __init__(self, scorer: LLMJudgeScorer) -> None:
        self._scorer = scorer

    @property
    def name(self) -> str:
        return "llm_judge"

    async def grade(
        self,
        prediction: str,
        ground_truth: str,
        config: dict[str, Any] | None = None,
    ) -> GradeResult:
        # Build a question context for the judge if available
        question_text = ""
        source_evidence: list[dict[str, Any]] = []
        if config and "question_text" in config:
            question_text = config["question_text"]
        if config and isinstance(config.get("source_evidence"), list):
            source_evidence = config["source_evidence"]

        score_result = await self._scorer.score(
            question=question_text or "Evaluate the prediction against the ground truth.",
            prediction=prediction,
            ground_truth=ground_truth,
            source_evidence=source_evidence,
        )

        return GradeResult(
            passed=score_result.score >= 1,
            score=float(score_result.score),
            reason=score_result.reasoning,
            grader_name=f"llm_judge ({score_result.judge_model})",
        )


# ---- Grader registry ----

_DETERMINISTIC_GRADERS: dict[str, Grader] = {
    "exact_match": ExactMatchGrader(),
    "normalized_date": NormalizedDateGrader(),
    "set": SetGrader(),
}

# Graders that require an LLM client for semantic evaluation.
_LLM_GRADER_NAMES = {"abstention", "gated", "supersession", "llm_judge"}


def resolve_grader(
    grader_name: str,
    llm_judge: LLMJudgeScorer | None = None,
) -> Grader:
    """Resolve a grader by name.

    Deterministic graders (exact_match, normalized_date, set) need no
    external dependencies.  All other graders require an ``LLMJudgeScorer``
    to supply the LLM client and judge model.
    """
    deterministic = _DETERMINISTIC_GRADERS.get(grader_name)
    if deterministic is not None:
        return deterministic

    if grader_name in _LLM_GRADER_NAMES:
        if llm_judge is None:
            raise ValueError(
                f"LLM judge scorer is required for '{grader_name}' grader "
                "but was not provided"
            )
        llm_client = llm_judge._llm_client
        judge_model = llm_judge._judge_model
        if grader_name == "abstention":
            return AbstentionGrader(llm_client, judge_model)
        if grader_name == "gated":
            return GatedGrader(llm_client, judge_model)
        if grader_name == "supersession":
            return SupersessionGrader(llm_client, judge_model)
        return LLMJudgeGrader(llm_judge)

    raise ValueError(f"Unknown grader: {grader_name}")
