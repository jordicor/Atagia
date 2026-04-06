"""Deterministic retrieval planning for candidate search."""

from __future__ import annotations

import calendar
from datetime import datetime, time, timedelta, tzinfo as datetime_tzinfo
import re

from atagia.core.clock import Clock
from atagia.memory.policy_manifest import ResolvedPolicy
from atagia.models.schemas_memory import (
    DetectedNeed,
    ExtractionConversationContext,
    MemoryScope,
    MemoryStatus,
    NeedTrigger,
    RetrievalPlan,
    TemporalQueryRange,
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "so",
        "that",
        "the",
        "this",
        "to",
        "was",
        "we",
        "with",
        "you",
        "your",
    }
)
# Expanded stopword list for question-shaped retrieval queries.
# Covers question words, auxiliaries, pronouns, articles, prepositions,
# conjunctions, and common filler words that pollute FTS5 AND queries.
_RETRIEVAL_STOPWORDS = frozenset(
    {
        # question words
        "what", "when", "where", "why", "who", "how",
        # auxiliaries
        "do", "does", "did", "has", "have", "had",
        "is", "are", "was", "were", "will", "would",
        "can", "could", "should",
        # pronouns
        "his", "her", "their", "its", "my", "your", "our",
        "he", "she", "they", "it", "i", "you", "we",
        # articles / prepositions / conjunctions
        "the", "a", "an", "in", "on", "at", "to", "for", "of",
        "with", "by", "from", "and", "or", "but", "not",
        "about", "into", "also", "both", "as", "so", "if",
        "than", "that", "this", "these", "those",
        # common filler
        "like", "just", "very", "really", "much", "some", "any",
        "all", "each", "every", "been", "being",
        # extras from _STOPWORDS not already covered
        "be", "me",
    }
)
_FTS_UNSAFE_PATTERN = re.compile(r"[^a-z0-9 ]")
_MAX_RETRIEVAL_QUERIES = 3
_ISO_DATE_PATTERN = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_RELATIVE_DAY_PATTERN = re.compile(r"\b(yesterday|today|tomorrow)\b", re.IGNORECASE)
_RELATIVE_SPAN_PATTERN = re.compile(r"\b(last|next)\s+(week|month|year)\b", re.IGNORECASE)
_MONTH_PATTERN = re.compile(
    r"\b("
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
    r")"
    r"(?:\s+(\d{4}))?\b",
    re.IGNORECASE,
)
_RELATIVE_DAY_OFFSETS = {
    "yesterday": -1,
    "today": 0,
    "tomorrow": 1,
}
_MONTH_NAME_TO_INDEX = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
_TEMPORAL_MONTH_PREFIXES = frozenset(
    {"in", "during", "throughout", "through", "for", "from", "since", "on", "by", "until", "before", "after"}
)
_PATTERN_KEYWORDS = (
    "prefer",
    "preference",
    "preferences",
    "pattern",
    "patterns",
    "usually",
    "often",
    "typically",
    "tend",
    "tendency",
    "tendencies",
    "habit",
    "habits",
    "style",
)
_ABSTRACT_MULTI_SESSION_KEYWORDS = (
    "across sessions",
    "across conversations",
    "across chats",
    "in general",
    "overall",
    "themes",
    "theme",
    "profile",
    "profiles",
    "multi-session",
    "multi session",
    "long term",
    "long-term",
    "recurring",
)
_TEMPORAL_HISTORY_KEYWORDS = ("history", "historical")


def _is_temporal_month_match(match: re.Match[str], message_text: str) -> bool:
    explicit_year = match.group(2)
    if explicit_year:
        return True
    prefix = message_text[: match.start()].rstrip().lower()
    if not prefix:
        return False
    previous_token = prefix.split()[-1].strip("([{\"'")
    return previous_token in _TEMPORAL_MONTH_PREFIXES


def _strip_temporal_month_expressions(message_text: str) -> str:
    sanitized_parts: list[str] = []
    last_index = 0
    for match in _MONTH_PATTERN.finditer(message_text):
        if not _is_temporal_month_match(match, message_text):
            continue
        sanitized_parts.append(message_text[last_index:match.start()])
        sanitized_parts.append(" ")
        last_index = match.end()
    sanitized_parts.append(message_text[last_index:])
    return "".join(sanitized_parts)


def _start_of_day(moment: datetime) -> datetime:
    return datetime.combine(moment.date(), time.min, tzinfo=moment.tzinfo)


def _end_of_day(moment: datetime) -> datetime:
    return datetime.combine(moment.date(), time.max, tzinfo=moment.tzinfo)


def _month_bounds(year: int, month: int, tzinfo: datetime_tzinfo | None) -> tuple[datetime, datetime]:
    start = datetime(year, month, 1, tzinfo=tzinfo)
    end = datetime(year, month, calendar.monthrange(year, month)[1], time.max.hour, time.max.minute, time.max.second, time.max.microsecond, tzinfo=tzinfo)
    return start, end


def _strip_temporal_expressions(message_text: str) -> str:
    sanitized = _ISO_DATE_PATTERN.sub(" ", message_text)
    sanitized = _RELATIVE_DAY_PATTERN.sub(" ", sanitized)
    sanitized = _RELATIVE_SPAN_PATTERN.sub(" ", sanitized)
    return _strip_temporal_month_expressions(sanitized)


def _relative_span_range(anchor: datetime, direction: str, unit: str) -> tuple[datetime, datetime]:
    step = -1 if direction == "last" else 1
    if unit == "week":
        current_week_start = _start_of_day(anchor) - timedelta(days=anchor.weekday())
        start = current_week_start + timedelta(days=7 * step)
        end = start + timedelta(days=6, hours=23, minutes=59, seconds=59, microseconds=999999)
        return start, end
    if unit == "month":
        month_offset = anchor.month + step
        year = anchor.year
        if month_offset == 0:
            month_offset = 12
            year -= 1
        elif month_offset == 13:
            month_offset = 1
            year += 1
        return _month_bounds(year, month_offset, anchor.tzinfo)
    year = anchor.year + step
    start = datetime(year, 1, 1, tzinfo=anchor.tzinfo)
    end = datetime(year, 12, 31, time.max.hour, time.max.minute, time.max.second, time.max.microsecond, tzinfo=anchor.tzinfo)
    return start, end


def _detected_temporal_intervals(message_text: str, anchor: datetime) -> list[tuple[datetime, datetime]]:
    intervals: list[tuple[datetime, datetime]] = []
    for year_text, month_text, day_text in _ISO_DATE_PATTERN.findall(message_text):
        moment = datetime(
            int(year_text),
            int(month_text),
            int(day_text),
            tzinfo=anchor.tzinfo,
        )
        intervals.append((_start_of_day(moment), _end_of_day(moment)))
    for relative_day in _RELATIVE_DAY_PATTERN.findall(message_text):
        offset = _RELATIVE_DAY_OFFSETS[relative_day.lower()]
        moment = anchor + timedelta(days=offset)
        intervals.append((_start_of_day(moment), _end_of_day(moment)))
    for direction, unit in _RELATIVE_SPAN_PATTERN.findall(message_text):
        intervals.append(_relative_span_range(anchor, direction.lower(), unit.lower()))
    for match in _MONTH_PATTERN.finditer(message_text):
        if not _is_temporal_month_match(match, message_text):
            continue
        month_name = match.group(1)
        explicit_year = match.group(2)
        month = _MONTH_NAME_TO_INDEX[month_name.lower()]
        year = int(explicit_year) if explicit_year else anchor.year
        intervals.append(_month_bounds(year, month, anchor.tzinfo))
    return intervals


def build_temporal_query_range(message_text: str, anchor: datetime) -> TemporalQueryRange | None:
    intervals = _detected_temporal_intervals(message_text, anchor)
    if not intervals:
        return None
    return TemporalQueryRange(
        start=min(start for start, _ in intervals),
        end=max(end for _, end in intervals),
    )


def build_retrieval_fts_queries(message_text: str) -> list[str]:
    """Generate multiple FTS5 queries from precise (AND) to broad (OR).

    Designed for retrieval over question-shaped natural language where a
    single AND query returns 0 hits because no document contains all terms.
    """
    sanitized_message_text = _strip_temporal_expressions(message_text)
    content_tokens: list[str] = []
    seen: set[str] = set()
    for token in _TOKEN_PATTERN.findall(sanitized_message_text.lower()):
        if token in seen or token in _RETRIEVAL_STOPWORDS:
            continue
        seen.add(token)
        content_tokens.append(token)

    if not content_tokens:
        return []

    # Sanitize each token for FTS5 safety
    sanitized: list[str] = []
    for token in content_tokens:
        clean = _FTS_UNSAFE_PATTERN.sub("", token).strip()
        if clean:
            sanitized.append(clean)
    if not sanitized:
        return []

    queries: list[str] = []

    if len(sanitized) >= 4:
        # AND with top 4 content tokens
        queries.append(" ".join(sanitized[:4]))
        # AND with top 3 content tokens
        queries.append(" ".join(sanitized[:3]))
        # OR with all content tokens
        queries.append(" OR ".join(sanitized))
    elif len(sanitized) == 3:
        # AND with all 3
        queries.append(" ".join(sanitized[:3]))
        # OR with all
        queries.append(" OR ".join(sanitized))
    else:
        # 1-2 tokens: single query (AND is same as listing them)
        queries.append(" ".join(sanitized))

    return queries[:_MAX_RETRIEVAL_QUERIES]


def build_safe_fts_queries(message_text: str) -> list[str]:
    """Build a single conservative FTS query for non-retrieval use cases.

    Used by ConsequenceChainBuilder and other modules that need a simple
    keyword query. For retrieval, use build_retrieval_fts_queries() instead.
    """
    content_tokens: list[str] = []
    seen: set[str] = set()
    for token in _TOKEN_PATTERN.findall(message_text.lower()):
        if token in seen or token in _STOPWORDS:
            continue
        seen.add(token)
        content_tokens.append(token)
    if not content_tokens:
        return []
    sanitized = [
        clean
        for token in content_tokens[:8]
        if (clean := _FTS_UNSAFE_PATTERN.sub("", token).strip())
    ]
    return [" ".join(sanitized)] if sanitized else []


_DEFAULT_SCOPE_ORDER = (
    MemoryScope.EPHEMERAL_SESSION,
    MemoryScope.CONVERSATION,
    MemoryScope.WORKSPACE,
    MemoryScope.ASSISTANT_MODE,
    MemoryScope.GLOBAL_USER,
)
_BROAD_SCOPE_ORDER = tuple(reversed(_DEFAULT_SCOPE_ORDER))
_NEED_BOOSTS: dict[NeedTrigger, float] = {
    NeedTrigger.AMBIGUITY: 1.2,
    NeedTrigger.CONTRADICTION: 1.35,
    NeedTrigger.FOLLOW_UP_FAILURE: 1.3,
    NeedTrigger.LOOP: 1.25,
    NeedTrigger.HIGH_STAKES: 1.25,
    NeedTrigger.MODE_SHIFT: 1.0,
    NeedTrigger.FRUSTRATION: 1.15,
    NeedTrigger.SENSITIVE_CONTEXT: 1.1,
    NeedTrigger.UNDER_SPECIFIED_REQUEST: 1.2,
}
assert set(_NEED_BOOSTS.keys()) == set(NeedTrigger), (
    f"_NEED_BOOSTS missing entries: {set(NeedTrigger) - set(_NEED_BOOSTS.keys())}"
)


class RetrievalPlanner:
    """Pure planner that converts policy and need signals into a retrieval plan."""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock

    def build_plan(
        self,
        message_text: str,
        conversation_context: ExtractionConversationContext,
        resolved_policy: ResolvedPolicy,
        detected_needs: list[DetectedNeed],
        cold_start: bool,
    ) -> RetrievalPlan:
        base_scope_filter = self._ordered_scopes(resolved_policy.allowed_scopes, _DEFAULT_SCOPE_ORDER)
        temporal_query_range = build_temporal_query_range(message_text, self._clock.now())
        plan = RetrievalPlan(
            assistant_mode_id=conversation_context.assistant_mode_id,
            workspace_id=conversation_context.workspace_id,
            conversation_id=conversation_context.conversation_id,
            fts_queries=self._build_fts_queries(message_text),
            scope_filter=base_scope_filter,
            status_filter=[MemoryStatus.ACTIVE],
            max_candidates=resolved_policy.retrieval_params.fts_limit,
            vector_limit=resolved_policy.retrieval_params.vector_limit,
            max_context_items=resolved_policy.retrieval_params.final_context_items,
            privacy_ceiling=resolved_policy.privacy_ceiling,
            retrieval_levels=self._determine_retrieval_levels(
                message_text,
                temporal_query_range=temporal_query_range,
            ),
            temporal_query_range=temporal_query_range,
            consequence_search_enabled=False,
            require_evidence_regrounding=False,
            need_driven_boosts={},
            skip_retrieval=cold_start,
        )

        sorted_needs = sorted(
            detected_needs,
            key=lambda need: (0 if need.need_type is NeedTrigger.MODE_SHIFT else 1, -need.confidence),
        )
        for need in sorted_needs:
            plan.need_driven_boosts[need.need_type] = _NEED_BOOSTS[need.need_type]

            if need.need_type is NeedTrigger.AMBIGUITY:
                plan.scope_filter = self._ordered_scopes(resolved_policy.allowed_scopes, _BROAD_SCOPE_ORDER)
                plan.max_candidates = self._increase_limit(plan.max_candidates)
                plan.vector_limit = self._increase_limit(plan.vector_limit)
                plan.consequence_search_enabled = True
            elif need.need_type is NeedTrigger.CONTRADICTION:
                if MemoryStatus.SUPERSEDED not in plan.status_filter:
                    plan.status_filter.append(MemoryStatus.SUPERSEDED)
            elif need.need_type is NeedTrigger.FOLLOW_UP_FAILURE:
                plan.scope_filter = self._prioritize_recent_conversation(plan.scope_filter)
                plan.consequence_search_enabled = True
            elif need.need_type is NeedTrigger.LOOP:
                plan.scope_filter = self._ordered_scopes(resolved_policy.allowed_scopes, _BROAD_SCOPE_ORDER)
                plan.max_candidates = self._increase_limit(plan.max_candidates)
                plan.vector_limit = self._increase_limit(plan.vector_limit)
                plan.consequence_search_enabled = True
            elif need.need_type is NeedTrigger.HIGH_STAKES:
                plan.max_candidates = self._increase_limit(plan.max_candidates)
                plan.vector_limit = self._increase_limit(plan.vector_limit)
                plan.consequence_search_enabled = True
                plan.require_evidence_regrounding = True
            elif need.need_type is NeedTrigger.MODE_SHIFT:
                plan.scope_filter = list(base_scope_filter)
                plan.status_filter = [MemoryStatus.ACTIVE]
                plan.max_candidates = resolved_policy.retrieval_params.fts_limit
                plan.vector_limit = resolved_policy.retrieval_params.vector_limit
                plan.max_context_items = resolved_policy.retrieval_params.final_context_items
                plan.privacy_ceiling = resolved_policy.privacy_ceiling
                plan.consequence_search_enabled = False
                plan.require_evidence_regrounding = False
            elif need.need_type is NeedTrigger.FRUSTRATION:
                plan.max_context_items = max(1, int(plan.max_context_items * 0.75))
            elif need.need_type is NeedTrigger.SENSITIVE_CONTEXT:
                plan.privacy_ceiling = min(plan.privacy_ceiling, 1)
            elif need.need_type is NeedTrigger.UNDER_SPECIFIED_REQUEST:
                plan.scope_filter = self._ordered_scopes(resolved_policy.allowed_scopes, _BROAD_SCOPE_ORDER)
                plan.max_candidates = self._increase_limit(plan.max_candidates)
                plan.vector_limit = self._increase_limit(plan.vector_limit)
                plan.consequence_search_enabled = True

        return plan

    @staticmethod
    def _build_fts_queries(message_text: str) -> list[str]:
        return build_retrieval_fts_queries(message_text)

    @staticmethod
    def _ordered_scopes(
        allowed_scopes: list[MemoryScope],
        preferred_order: tuple[MemoryScope, ...],
    ) -> list[MemoryScope]:
        allowed = set(allowed_scopes)
        return [scope for scope in preferred_order if scope in allowed]

    @staticmethod
    def _increase_limit(limit: int) -> int:
        return max(limit + 1, int(limit * 1.5))

    @staticmethod
    def _prioritize_recent_conversation(scopes: list[MemoryScope]) -> list[MemoryScope]:
        preferred = [MemoryScope.EPHEMERAL_SESSION, MemoryScope.CONVERSATION]
        prioritized = [scope for scope in preferred if scope in scopes]
        prioritized.extend(scope for scope in scopes if scope not in prioritized)
        return prioritized

    @staticmethod
    def _determine_retrieval_levels(
        message_text: str,
        *,
        temporal_query_range: TemporalQueryRange | None,
    ) -> list[int]:
        normalized = f" {message_text.lower()} "
        if temporal_query_range is not None or any(keyword in normalized for keyword in _TEMPORAL_HISTORY_KEYWORDS):
            return [0, 1]
        if any(keyword in normalized for keyword in _ABSTRACT_MULTI_SESSION_KEYWORDS):
            return [2, 1, 0]
        if any(keyword in normalized for keyword in _PATTERN_KEYWORDS):
            return [1, 0]
        return [0]
