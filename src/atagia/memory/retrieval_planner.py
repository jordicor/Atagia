"""Deterministic retrieval planning for candidate search."""

from __future__ import annotations

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


def build_retrieval_fts_queries(message_text: str) -> list[str]:
    """Generate multiple FTS5 queries from precise (AND) to broad (OR).

    Designed for retrieval over question-shaped natural language where a
    single AND query returns 0 hits because no document contains all terms.
    """
    content_tokens: list[str] = []
    seen: set[str] = set()
    for token in _TOKEN_PATTERN.findall(message_text.lower()):
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
        plan = RetrievalPlan(
            assistant_mode_id=conversation_context.assistant_mode_id,
            workspace_id=conversation_context.workspace_id,
            conversation_id=conversation_context.conversation_id,
            fts_queries=self._build_fts_queries(message_text),
            scope_filter=base_scope_filter,
            status_filter=[MemoryStatus.ACTIVE],
            max_candidates=resolved_policy.retrieval_params.fts_limit,
            max_context_items=resolved_policy.retrieval_params.final_context_items,
            privacy_ceiling=resolved_policy.privacy_ceiling,
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
            elif need.need_type is NeedTrigger.CONTRADICTION:
                if MemoryStatus.SUPERSEDED not in plan.status_filter:
                    plan.status_filter.append(MemoryStatus.SUPERSEDED)
            elif need.need_type is NeedTrigger.FOLLOW_UP_FAILURE:
                plan.scope_filter = self._prioritize_recent_conversation(plan.scope_filter)
            elif need.need_type is NeedTrigger.LOOP:
                plan.scope_filter = self._ordered_scopes(resolved_policy.allowed_scopes, _BROAD_SCOPE_ORDER)
                plan.max_candidates = self._increase_limit(plan.max_candidates)
            elif need.need_type is NeedTrigger.HIGH_STAKES:
                plan.max_candidates = self._increase_limit(plan.max_candidates)
                plan.require_evidence_regrounding = True
            elif need.need_type is NeedTrigger.MODE_SHIFT:
                plan.scope_filter = list(base_scope_filter)
                plan.status_filter = [MemoryStatus.ACTIVE]
                plan.max_candidates = resolved_policy.retrieval_params.fts_limit
                plan.max_context_items = resolved_policy.retrieval_params.final_context_items
                plan.privacy_ceiling = resolved_policy.privacy_ceiling
                plan.require_evidence_regrounding = False
            elif need.need_type is NeedTrigger.FRUSTRATION:
                plan.max_context_items = max(1, int(plan.max_context_items * 0.75))
            elif need.need_type is NeedTrigger.SENSITIVE_CONTEXT:
                plan.privacy_ceiling = min(plan.privacy_ceiling, 1)
            elif need.need_type is NeedTrigger.UNDER_SPECIFIED_REQUEST:
                plan.scope_filter = self._ordered_scopes(resolved_policy.allowed_scopes, _BROAD_SCOPE_ORDER)
                plan.max_candidates = self._increase_limit(plan.max_candidates)

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
