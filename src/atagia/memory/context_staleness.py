"""Deterministic staleness scoring for adaptive context cache entries."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from atagia.core.clock import Clock, SystemClock
from atagia.memory.policy_manifest import ResolvedPolicy
from atagia.models.schemas_cache import ContextCacheEntry

MESSAGE_PENALTY_PER_MESSAGE = 0.1
TIME_PENALTY_PER_MINUTE = 0.05
PACE_LABEL_MULTIPLIER = {
    "fast": 1.25,
    "default": 1.0,
    "methodical": 0.8,
}
SHORT_FOLLOWUP_PHRASES = frozenset(
    {
        "ok",
        "okay",
        "continue",
        "go on",
        "carry on",
        "go ahead",
        "keep going",
        "please continue",
        "sounds good",
        "thanks",
        "thank you",
        "tell me more",
        "more",
    }
)
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "about",
        "as",
        "at",
        "be",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "me",
        "my",
        "now",
        "of",
        "on",
        "or",
        "our",
        "please",
        "should",
        "so",
        "that",
        "the",
        "then",
        "this",
        "to",
        "we",
        "what",
        "when",
        "where",
        "which",
        "why",
        "with",
        "would",
        "you",
        "your",
    }
)
LOW_CONTEXT_TOKENS = frozenset({"this", "that", "it", "those", "these", "there", "here"})
CONTRADICTION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bactually\b",
        r"\bnot anymore\b",
        r"\bthat changed\b",
        r"\binstead\b",
        r"\bno longer\b",
        r"\bi changed my mind\b",
        r"\bcorrection\b",
    )
)
HIGH_STAKES_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bmedical\b",
        r"\bdoctor\b",
        r"\bmedication\b",
        r"\bdosage\b",
        r"\blegal\b",
        r"\blawyer\b",
        r"\blawsuit\b",
        r"\bfinancial\b",
        r"\binvest(?:ing|ment)?\b",
        r"\btax(?:es)?\b",
        r"\birs\b",
        r"\bsafety\b",
        r"\bself[- ]?harm\b",
        r"\bsuicid(?:e|al)\b",
        r"\bemergency\b",
    )
)
SENSITIVE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\btrauma\b",
        r"\babuse\b",
        r"\bdiagnos(?:is|ed|es)\b",
        r"\bfinances?\b",
        r"\bpasswords?\b",
        r"\bsecret(?:s)?\b",
        r"\bprivate account\b",
        r"\bbank account\b",
        r"\bsocial security\b",
        r"\bssn\b",
    )
)
TOKEN_RE = re.compile(r"[a-z0-9']+")
MODE_KEYWORDS = {
    "coding_debug": frozenset({"coding", "debug", "debugging"}),
    "research_deep_dive": frozenset({"research", "deep", "paper", "papers"}),
    "companion": frozenset({"companion"}),
    "brainstorm": frozenset({"brainstorm", "brainstorming"}),
    "biographical_interview": frozenset({"biographical", "interview", "chronology"}),
    "general_qa": frozenset({"general", "qa"}),
}


class ContextStalenessRequest(BaseModel):
    """Inputs required to evaluate whether a cache entry is still safe to reuse."""

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    workspace_id: str | None = None
    message_text: str = Field(min_length=1)
    current_message_seq: int = Field(ge=0)
    cache_enabled: bool = True
    benchmark_mode: bool = False
    replay_mode: bool = False
    evaluation_mode: bool = False
    mcp_mode: bool = False

    @field_validator("message_text")
    @classmethod
    def validate_message_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("message_text must be non-empty")
        return normalized


class ContextStalenessScore(BaseModel):
    """Deterministic staleness decision plus debug metadata."""

    model_config = ConfigDict(extra="forbid")

    staleness: float = Field(ge=0.0, le=1.0)
    should_refresh: bool
    hard_sync: bool
    effective_sync_threshold: float = Field(gt=0.0, le=1.0)
    messages_since_refresh: int = Field(ge=0)
    minutes_since_refresh: float = Field(ge=0.0)
    effective_max_messages_without_refresh: int = Field(ge=1)
    effective_max_minutes_without_refresh: int = Field(ge=1)
    pace_label: str
    pace_multiplier: float = Field(gt=0.0)
    message_penalty: float = Field(ge=0.0, le=1.0)
    time_penalty: float = Field(ge=0.0, le=1.0)
    topic_penalty: float = Field(ge=0.0, le=1.0)
    safety_penalty: float = Field(ge=0.0, le=1.0)
    token_overlap_ratio: float = Field(ge=0.0, le=1.0)
    short_followup: bool
    matched_signals: list[str] = Field(default_factory=list)


class ContextStalenessScorer:
    """Conservative, deterministic scorer for cache-entry reuse safety."""

    def __init__(self, clock: Clock | None = None) -> None:
        self._clock = clock or SystemClock()

    def score(
        self,
        entry: ContextCacheEntry | dict[str, Any],
        request: ContextStalenessRequest | dict[str, Any],
        resolved_policy: ResolvedPolicy,
    ) -> ContextStalenessScore:
        request_model = ContextStalenessRequest.model_validate(request)
        try:
            cache_entry = ContextCacheEntry.model_validate(entry)
        except ValidationError:
            return self._hard_sync_score(
                resolved_policy=resolved_policy,
                matched_signals=["cache_entry_validation_failed"],
            )

        pace_label, pace_multiplier = _resolve_pace_multiplier(cache_entry.contract)
        cache_policy = resolved_policy.context_cache_policy
        effective_max_messages = max(
            1,
            round(cache_policy.max_messages_without_refresh * pace_multiplier),
        )
        effective_max_minutes = max(
            1,
            round(cache_policy.max_minutes_without_refresh * pace_multiplier),
        )

        hard_signal = self._validate_cache_entry(
            cache_entry=cache_entry,
            request=request_model,
            resolved_policy=resolved_policy,
        )
        if hard_signal is not None:
            return self._hard_sync_score(
                resolved_policy=resolved_policy,
                matched_signals=[hard_signal],
                pace_label=pace_label,
                pace_multiplier=pace_multiplier,
                effective_max_messages=effective_max_messages,
                effective_max_minutes=effective_max_minutes,
            )

        cached_at = _parse_cached_at(cache_entry.cached_at)
        if cached_at is None:
            return self._hard_sync_score(
                resolved_policy=resolved_policy,
                matched_signals=["cached_at_invalid"],
                pace_label=pace_label,
                pace_multiplier=pace_multiplier,
                effective_max_messages=effective_max_messages,
                effective_max_minutes=effective_max_minutes,
            )

        messages_since_refresh = max(
            0,
            request_model.current_message_seq - cache_entry.last_retrieval_message_seq,
        )
        minutes_since_refresh = max(
            0.0,
            (self._clock.now() - cached_at).total_seconds() / 60.0,
        )
        if messages_since_refresh > effective_max_messages:
            return self._hard_sync_score(
                resolved_policy=resolved_policy,
                matched_signals=["message_ceiling_exceeded"],
                pace_label=pace_label,
                pace_multiplier=pace_multiplier,
                effective_max_messages=effective_max_messages,
                effective_max_minutes=effective_max_minutes,
                messages_since_refresh=messages_since_refresh,
                minutes_since_refresh=minutes_since_refresh,
            )
        if minutes_since_refresh > effective_max_minutes:
            return self._hard_sync_score(
                resolved_policy=resolved_policy,
                matched_signals=["time_ceiling_exceeded"],
                pace_label=pace_label,
                pace_multiplier=pace_multiplier,
                effective_max_messages=effective_max_messages,
                effective_max_minutes=effective_max_minutes,
                messages_since_refresh=messages_since_refresh,
                minutes_since_refresh=minutes_since_refresh,
            )

        normalized_message = _normalize_message(request_model.message_text)
        short_followup = normalized_message in SHORT_FOLLOWUP_PHRASES
        token_overlap_ratio = _token_overlap_ratio(
            request_model.message_text,
            cache_entry.last_user_message_text,
        )
        message_penalty = _clamp(messages_since_refresh * MESSAGE_PENALTY_PER_MESSAGE)
        time_penalty = _clamp(minutes_since_refresh * TIME_PENALTY_PER_MINUTE)
        topic_penalty = _topic_penalty(
            token_overlap_ratio=token_overlap_ratio,
            short_followup=short_followup,
        )
        matched_signals: list[str] = []
        if short_followup:
            matched_signals.append("short_followup")
        if token_overlap_ratio >= 0.6:
            matched_signals.append("high_token_overlap")
        elif token_overlap_ratio <= 0.1 and not short_followup:
            matched_signals.append("low_token_overlap")

        contradiction = _matches_any(request_model.message_text, CONTRADICTION_PATTERNS)
        if contradiction:
            return self._hard_sync_score(
                resolved_policy=resolved_policy,
                matched_signals=[*matched_signals, "contradiction_language"],
                pace_label=pace_label,
                pace_multiplier=pace_multiplier,
                effective_max_messages=effective_max_messages,
                effective_max_minutes=effective_max_minutes,
                messages_since_refresh=messages_since_refresh,
                minutes_since_refresh=minutes_since_refresh,
                message_penalty=message_penalty,
                time_penalty=time_penalty,
                topic_penalty=topic_penalty,
                token_overlap_ratio=token_overlap_ratio,
                short_followup=short_followup,
            )

        mode_shift = _has_explicit_mode_shift(
            message_text=request_model.message_text,
            current_mode_id=resolved_policy.assistant_mode_id.value,
        )
        if mode_shift:
            return self._hard_sync_score(
                resolved_policy=resolved_policy,
                matched_signals=[*matched_signals, "mode_shift_language"],
                pace_label=pace_label,
                pace_multiplier=pace_multiplier,
                effective_max_messages=effective_max_messages,
                effective_max_minutes=effective_max_minutes,
                messages_since_refresh=messages_since_refresh,
                minutes_since_refresh=minutes_since_refresh,
                message_penalty=message_penalty,
                time_penalty=time_penalty,
                topic_penalty=topic_penalty,
                token_overlap_ratio=token_overlap_ratio,
                short_followup=short_followup,
            )

        if _matches_any(request_model.message_text, HIGH_STAKES_PATTERNS):
            return self._hard_sync_score(
                resolved_policy=resolved_policy,
                matched_signals=[*matched_signals, "high_stakes_language"],
                pace_label=pace_label,
                pace_multiplier=pace_multiplier,
                effective_max_messages=effective_max_messages,
                effective_max_minutes=effective_max_minutes,
                messages_since_refresh=messages_since_refresh,
                minutes_since_refresh=minutes_since_refresh,
                message_penalty=message_penalty,
                time_penalty=time_penalty,
                topic_penalty=topic_penalty,
                token_overlap_ratio=token_overlap_ratio,
                short_followup=short_followup,
            )

        if _matches_any(request_model.message_text, SENSITIVE_PATTERNS):
            return self._hard_sync_score(
                resolved_policy=resolved_policy,
                matched_signals=[*matched_signals, "sensitive_language"],
                pace_label=pace_label,
                pace_multiplier=pace_multiplier,
                effective_max_messages=effective_max_messages,
                effective_max_minutes=effective_max_minutes,
                messages_since_refresh=messages_since_refresh,
                minutes_since_refresh=minutes_since_refresh,
                message_penalty=message_penalty,
                time_penalty=time_penalty,
                topic_penalty=topic_penalty,
                token_overlap_ratio=token_overlap_ratio,
                short_followup=short_followup,
            )

        safety_penalty = 0.0
        if _is_ambiguous_wording(
            message_text=request_model.message_text,
            short_followup=short_followup,
            token_overlap_ratio=token_overlap_ratio,
        ):
            matched_signals.append("ambiguous_wording")
            safety_penalty = max(safety_penalty, cache_policy.sync_threshold)

        staleness = _clamp(
            message_penalty + time_penalty + topic_penalty + safety_penalty,
        )
        return ContextStalenessScore(
            staleness=staleness,
            should_refresh=staleness >= cache_policy.sync_threshold,
            hard_sync=False,
            effective_sync_threshold=cache_policy.sync_threshold,
            messages_since_refresh=messages_since_refresh,
            minutes_since_refresh=round(minutes_since_refresh, 4),
            effective_max_messages_without_refresh=effective_max_messages,
            effective_max_minutes_without_refresh=effective_max_minutes,
            pace_label=pace_label,
            pace_multiplier=pace_multiplier,
            message_penalty=message_penalty,
            time_penalty=time_penalty,
            topic_penalty=topic_penalty,
            safety_penalty=safety_penalty,
            token_overlap_ratio=token_overlap_ratio,
            short_followup=short_followup,
            matched_signals=_unique(matched_signals),
        )

    def _validate_cache_entry(
        self,
        *,
        cache_entry: ContextCacheEntry,
        request: ContextStalenessRequest,
        resolved_policy: ResolvedPolicy,
    ) -> str | None:
        if not request.cache_enabled:
            return "cache_disabled"
        if request.benchmark_mode:
            return "benchmark_mode"
        if request.replay_mode:
            return "replay_mode"
        if request.evaluation_mode:
            return "evaluation_mode"
        if request.mcp_mode:
            return "mcp_mode"
        if cache_entry.user_id != request.user_id:
            return "user_id_mismatch"
        if cache_entry.conversation_id != request.conversation_id:
            return "conversation_id_mismatch"
        if cache_entry.assistant_mode_id != resolved_policy.assistant_mode_id.value:
            return "assistant_mode_id_mismatch"
        if cache_entry.workspace_id != request.workspace_id:
            return "workspace_id_mismatch"
        if cache_entry.policy_prompt_hash != resolved_policy.prompt_hash:
            return "policy_prompt_hash_mismatch"
        if request.current_message_seq < cache_entry.last_retrieval_message_seq:
            return "message_sequence_rewind"
        return None

    def _hard_sync_score(
        self,
        *,
        resolved_policy: ResolvedPolicy,
        matched_signals: list[str],
        pace_label: str = "default",
        pace_multiplier: float = 1.0,
        effective_max_messages: int | None = None,
        effective_max_minutes: int | None = None,
        messages_since_refresh: int = 0,
        minutes_since_refresh: float = 0.0,
        message_penalty: float = 0.0,
        time_penalty: float = 0.0,
        topic_penalty: float = 0.0,
        token_overlap_ratio: float = 0.0,
        short_followup: bool = False,
    ) -> ContextStalenessScore:
        policy = resolved_policy.context_cache_policy
        return ContextStalenessScore(
            staleness=1.0,
            should_refresh=True,
            hard_sync=True,
            effective_sync_threshold=policy.sync_threshold,
            messages_since_refresh=messages_since_refresh,
            minutes_since_refresh=round(minutes_since_refresh, 4),
            effective_max_messages_without_refresh=(
                effective_max_messages or policy.max_messages_without_refresh
            ),
            effective_max_minutes_without_refresh=(
                effective_max_minutes or policy.max_minutes_without_refresh
            ),
            pace_label=pace_label,
            pace_multiplier=pace_multiplier,
            message_penalty=message_penalty,
            time_penalty=time_penalty,
            topic_penalty=topic_penalty,
            safety_penalty=1.0,
            token_overlap_ratio=token_overlap_ratio,
            short_followup=short_followup,
            matched_signals=_unique(matched_signals),
        )


def _resolve_pace_multiplier(contract: dict[str, dict[str, Any]]) -> tuple[str, float]:
    pace_payload = contract.get("pace") or {}
    pace_label = str(pace_payload.get("label", "default")).strip().lower() or "default"
    return pace_label, PACE_LABEL_MULTIPLIER.get(pace_label, 1.0)


def _parse_cached_at(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _normalize_message(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", text.lower())).strip()


def _tokenize(text: str, *, drop_stopwords: bool) -> list[str]:
    tokens = TOKEN_RE.findall(text.lower())
    if not drop_stopwords:
        return tokens
    filtered = [token for token in tokens if token not in STOPWORDS]
    return filtered or tokens


def _token_overlap_ratio(current_text: str, previous_text: str) -> float:
    current_tokens = set(_tokenize(current_text, drop_stopwords=True))
    previous_tokens = set(_tokenize(previous_text, drop_stopwords=True))
    if not current_tokens or not previous_tokens:
        return 0.0
    overlap = current_tokens & previous_tokens
    return _clamp(len(overlap) / len(current_tokens))


def _topic_penalty(*, token_overlap_ratio: float, short_followup: bool) -> float:
    if short_followup:
        return 0.0
    if token_overlap_ratio >= 0.6:
        return 0.0
    if token_overlap_ratio >= 0.3:
        return 0.15
    if token_overlap_ratio > 0.0:
        return 0.3
    return 0.5


def _matches_any(text: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(pattern.search(text) is not None for pattern in patterns)


def _has_explicit_mode_shift(message_text: str, current_mode_id: str) -> bool:
    normalized = _normalize_message(message_text)
    if not any(keyword in normalized for keyword in ("switch", "change", "instead", "rather", "mode")):
        return False
    for mode_id, keywords in MODE_KEYWORDS.items():
        if mode_id == current_mode_id:
            continue
        if any(keyword in normalized for keyword in keywords):
            return True
    if "different mode" in normalized or "switch mode" in normalized or "change mode" in normalized:
        return True
    if "instead of" in normalized or "rather than" in normalized:
        return True
    return False


def _is_ambiguous_wording(
    *,
    message_text: str,
    short_followup: bool,
    token_overlap_ratio: float,
) -> bool:
    if short_followup:
        return False
    normalized = _normalize_message(message_text)
    tokens = _tokenize(message_text, drop_stopwords=False)
    if not tokens:
        return True
    if len(tokens) > 6:
        return False
    if token_overlap_ratio >= 0.2:
        return False
    if "?" in message_text:
        return True
    if any(token in LOW_CONTEXT_TOKENS for token in tokens):
        return True
    if normalized in {"thoughts", "what now", "and then", "and now"}:
        return True
    return False


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
