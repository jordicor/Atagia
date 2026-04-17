"""Deterministic retrieval planning for candidate search."""

from __future__ import annotations

import re

from atagia.memory.policy_manifest import ResolvedPolicy
from atagia.models.schemas_memory import (
    ExtractionConversationContext,
    MemoryScope,
    MemoryStatus,
    NeedTrigger,
    PlannedSubQuery,
    QueryIntelligenceResult,
    RetrievalPlan,
    SparseQueryHint,
)

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
_FTS5_OPERATORS = frozenset({"and", "or", "not", "near"})
_MAX_RETRIEVAL_QUERIES = 3
_MAX_HINT_QUERY_TOKENS = 6

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


def build_retrieval_fts_queries(
    message_text: str,
    *,
    quoted_phrases: list[str] | None = None,
    must_keep_terms: list[str] | None = None,
) -> list[str]:
    """Generate multiple FTS5 queries from precise (AND) to broad (OR)."""
    content_tokens = _collect_content_tokens(message_text)
    normalized_quoted_phrases = _normalize_quoted_phrases(quoted_phrases or [])
    normalized_must_keep_terms = _collect_tokens_from_values(must_keep_terms or [])

    if not normalized_quoted_phrases and not normalized_must_keep_terms:
        return _build_default_fts_queries(content_tokens, original_text=message_text)

    content_tokens = _stable_token_union(content_tokens, normalized_must_keep_terms)
    if not content_tokens and not normalized_quoted_phrases:
        raise ValueError(
            f"Sub-query produced no searchable FTS tokens: {message_text!r}"
        )

    queries: list[str] = []
    seen_queries: set[str] = set()
    for phrase in normalized_quoted_phrases:
        quoted_query = f'"{phrase}"'
        if quoted_query in seen_queries:
            continue
        seen_queries.add(quoted_query)
        queries.append(quoted_query)

    if content_tokens:
        anchor_tokens = _stable_token_union(
            normalized_must_keep_terms,
            _collect_tokens_from_values(normalized_quoted_phrases),
        )
        if anchor_tokens and normalized_must_keep_terms and not normalized_quoted_phrases:
            anchor_first_query = " ".join(
                _stable_token_union(anchor_tokens, content_tokens)[:_MAX_HINT_QUERY_TOKENS]
            )
            if anchor_first_query not in seen_queries:
                seen_queries.add(anchor_first_query)
                queries.append(anchor_first_query)

        strong_query = " ".join(content_tokens[:_MAX_HINT_QUERY_TOKENS])
        if strong_query not in seen_queries:
            seen_queries.add(strong_query)
            queries.append(strong_query)

        if len(content_tokens) >= 3:
            broad_query = " OR ".join(content_tokens)
            if broad_query not in seen_queries:
                seen_queries.add(broad_query)
                queries.append(broad_query)

    return queries[:_MAX_RETRIEVAL_QUERIES]


def _build_default_fts_queries(
    content_tokens: list[str],
    *,
    original_text: str,
) -> list[str]:
    if not content_tokens:
        raise ValueError(
            f"Sub-query produced no searchable FTS tokens: {original_text!r}"
        )

    queries: list[str] = []
    if len(content_tokens) >= 4:
        queries.append(" ".join(content_tokens[:4]))
        queries.append(" ".join(content_tokens[:3]))
        queries.append(" OR ".join(content_tokens))
    elif len(content_tokens) == 3:
        queries.append(" ".join(content_tokens[:3]))
        queries.append(" OR ".join(content_tokens))
    else:
        queries.append(" ".join(content_tokens))
    return queries[:_MAX_RETRIEVAL_QUERIES]


def build_safe_fts_queries(message_text: str) -> list[str]:
    """Build a single conservative FTS query for non-retrieval use cases."""
    content_tokens = _collect_content_tokens(message_text)
    if not content_tokens:
        return []
    return [" ".join(content_tokens[:8])]


def _collect_content_tokens(message_text: str) -> list[str]:
    return _collect_tokens_from_values([message_text])


def _collect_tokens_from_values(values: list[str]) -> list[str]:
    content_tokens: list[str] = []
    seen: set[str] = set()
    for value in values:
        for token in _TOKEN_PATTERN.findall(value.lower()):
            clean = _sanitize_fts_token(token)
            if clean is None or clean in seen:
                continue
            seen.add(clean)
            content_tokens.append(clean)
    return content_tokens


def _normalize_quoted_phrases(values: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        tokens = _collect_content_tokens(value)
        if len(tokens) < 2:
            continue
        phrase = " ".join(tokens)
        if phrase in seen:
            continue
        seen.add(phrase)
        normalized.append(phrase)
    return normalized


def _stable_token_union(primary: list[str], secondary: list[str]) -> list[str]:
    merged = list(primary)
    seen = set(primary)
    for token in secondary:
        if token in seen:
            continue
        seen.add(token)
        merged.append(token)
    return merged


def _resolve_sparse_phrase(
    sub_query: str,
    sparse_hint: SparseQueryHint | None,
) -> str:
    if sparse_hint is None:
        return sub_query
    if sparse_hint.fts_phrase is not None:
        return sparse_hint.fts_phrase
    hint_tokens = _stable_token_union(
        _collect_tokens_from_values(list(sparse_hint.quoted_phrases)),
        _collect_tokens_from_values(list(sparse_hint.must_keep_terms)),
    )
    if hint_tokens:
        return " ".join(hint_tokens[:_MAX_HINT_QUERY_TOKENS])
    return sub_query


def _sanitize_fts_token(token: str) -> str | None:
    normalized = token.strip("_")
    if not normalized:
        return None
    if not any(character.isalnum() for character in normalized):
        return None
    if normalized.lower() in _FTS5_OPERATORS:
        return None
    return normalized


class RetrievalPlanner:
    """Pure planner that converts query intelligence into a retrieval plan."""

    def build_plan(
        self,
        *,
        original_query: str,
        query_intelligence: QueryIntelligenceResult,
        conversation_context: ExtractionConversationContext,
        resolved_policy: ResolvedPolicy,
        cold_start: bool,
    ) -> RetrievalPlan:
        base_scope_filter = self._ordered_scopes(
            resolved_policy.allowed_scopes,
            _DEFAULT_SCOPE_ORDER,
        )
        sub_query_plans = self._build_sub_query_plans(
            query_intelligence.sub_queries,
            query_intelligence.sparse_query_hints,
        )
        plan = RetrievalPlan(
            original_query=original_query,
            assistant_mode_id=conversation_context.assistant_mode_id,
            workspace_id=conversation_context.workspace_id,
            conversation_id=conversation_context.conversation_id,
            fts_queries=self._flatten_fts_queries(sub_query_plans),
            sub_query_plans=sub_query_plans,
            callback_bias=query_intelligence.callback_bias,
            raw_context_access_mode=query_intelligence.raw_context_access_mode,
            query_type=query_intelligence.query_type,
            scope_filter=base_scope_filter,
            status_filter=[MemoryStatus.ACTIVE],
            max_candidates=resolved_policy.retrieval_params.fts_limit,
            vector_limit=resolved_policy.retrieval_params.vector_limit,
            max_context_items=resolved_policy.retrieval_params.final_context_items,
            privacy_ceiling=resolved_policy.privacy_ceiling,
            retrieval_levels=list(query_intelligence.retrieval_levels),
            temporal_query_range=query_intelligence.temporal_range,
            consequence_search_enabled=False,
            require_evidence_regrounding=False,
            need_driven_boosts={},
            skip_retrieval=cold_start,
            exact_recall_mode=bool(query_intelligence.exact_recall_needed),
            exact_facets=list(query_intelligence.exact_facets),
        )

        sorted_needs = sorted(
            query_intelligence.needs,
            key=lambda need: (
                0 if need.need_type is NeedTrigger.MODE_SHIFT else 1,
                -need.confidence,
            ),
        )
        for need in sorted_needs:
            plan.need_driven_boosts[need.need_type] = _NEED_BOOSTS[need.need_type]

            if need.need_type is NeedTrigger.AMBIGUITY:
                plan.scope_filter = self._ordered_scopes(
                    resolved_policy.allowed_scopes,
                    _BROAD_SCOPE_ORDER,
                )
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
                plan.scope_filter = self._ordered_scopes(
                    resolved_policy.allowed_scopes,
                    _BROAD_SCOPE_ORDER,
                )
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
                plan.scope_filter = self._ordered_scopes(
                    resolved_policy.allowed_scopes,
                    _BROAD_SCOPE_ORDER,
                )
                plan.max_candidates = self._increase_limit(plan.max_candidates)
                plan.vector_limit = self._increase_limit(plan.vector_limit)
                plan.consequence_search_enabled = True

        return plan

    @staticmethod
    def _build_sub_query_plans(
        sub_queries: list[str],
        sparse_query_hints: list[SparseQueryHint],
    ) -> list[PlannedSubQuery]:
        sparse_hints_by_sub_query = {
            hint.sub_query_text: hint for hint in sparse_query_hints
        }
        sub_query_plans: list[PlannedSubQuery] = []
        for sub_query in sub_queries:
            sparse_hint = sparse_hints_by_sub_query.get(sub_query)
            sparse_phrase = _resolve_sparse_phrase(sub_query, sparse_hint)
            quoted_phrases = list(sparse_hint.quoted_phrases) if sparse_hint is not None else []
            must_keep_terms = list(sparse_hint.must_keep_terms) if sparse_hint is not None else []
            fts_queries = build_retrieval_fts_queries(
                sparse_phrase,
                quoted_phrases=quoted_phrases,
                must_keep_terms=must_keep_terms,
            )
            if not fts_queries:
                raise ValueError(f"Sub-query did not produce any FTS rewrites: {sub_query!r}")
            sub_query_plans.append(
                PlannedSubQuery(
                    text=sub_query,
                    sparse_phrase=sparse_phrase,
                    quoted_phrases=quoted_phrases,
                    must_keep_terms=must_keep_terms,
                    fts_queries=fts_queries,
                )
            )
        return sub_query_plans

    @staticmethod
    def _flatten_fts_queries(sub_query_plans: list[PlannedSubQuery]) -> list[str]:
        flattened: list[str] = []
        seen: set[str] = set()
        for sub_query in sub_query_plans:
            for query in sub_query.fts_queries:
                if query in seen:
                    continue
                seen.add(query)
                flattened.append(query)
        return flattened

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
