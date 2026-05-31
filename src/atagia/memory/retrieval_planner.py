"""Deterministic retrieval planning for candidate search."""

from __future__ import annotations

from dataclasses import dataclass
import re

from atagia.memory.policy_manifest import ResolvedRetrievalPolicy
from atagia.models.schemas_memory import (
    ExtractionConversationContext,
    MemoryScope,
    MemoryStatus,
    NeedTrigger,
    PlannedSubQuery,
    QueryIntelligenceResult,
    RetrievalPlan,
    RuntimeAnchor,
    SparseQueryHint,
)

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
_FTS5_OPERATORS = frozenset({"and", "or", "not", "near"})
_MAX_RETRIEVAL_QUERIES = 3
_MAX_EXACT_RECALL_RETRIEVAL_QUERIES = 5
_MAX_HINT_QUERY_TOKENS = 6
_MAX_ANCHORS_WITH_ALIAS_QUERIES = 2
_MAX_ALIASES_PER_ANCHOR = 2
_MAX_ALIAS_QUERY_SPECS_PER_SUBQUERY = 6
_MAX_PERSON_ANCHOR_BACKOFF_TOKENS = 3
_MIN_PERSON_ANCHOR_BACKOFF_TOKEN_LENGTH = 3
_EXACT_BROAD_LIST_MIN_CANDIDATES = 90

_DEFAULT_SCOPE_ORDER = (
    MemoryScope.EPHEMERAL_SESSION,
    MemoryScope.CONVERSATION,
    MemoryScope.WORKSPACE,
    MemoryScope.GLOBAL_USER,
)
_BROAD_SCOPE_ORDER = tuple(reversed(_DEFAULT_SCOPE_ORDER))


@dataclass(frozen=True)
class RetrievalFtsQuerySpec:
    """Materialized FTS query plus a behavior-neutral diagnostic label."""

    query: str
    kind: str


def build_retrieval_fts_queries(
    message_text: str,
    *,
    quoted_phrases: list[str] | None = None,
    must_keep_terms: list[str] | None = None,
    exact_recall: bool = False,
) -> list[str]:
    """Generate multiple FTS5 queries from precise (AND) to broad (OR)."""
    return [
        spec.query
        for spec in build_retrieval_fts_query_specs(
            message_text,
            quoted_phrases=quoted_phrases,
            must_keep_terms=must_keep_terms,
            exact_recall=exact_recall,
        )
    ]


def build_retrieval_fts_query_specs(
    message_text: str,
    *,
    quoted_phrases: list[str] | None = None,
    must_keep_terms: list[str] | None = None,
    exact_recall: bool = False,
) -> list[RetrievalFtsQuerySpec]:
    """Generate FTS5 queries and label how each rewrite was materialized."""
    content_tokens = _collect_content_tokens(message_text)
    normalized_quoted_phrases = _normalize_quoted_phrases(quoted_phrases or [])
    normalized_must_keep_terms = _collect_tokens_from_values(must_keep_terms or [])
    max_queries = (
        _MAX_EXACT_RECALL_RETRIEVAL_QUERIES
        if exact_recall
        else _MAX_RETRIEVAL_QUERIES
    )

    if not normalized_quoted_phrases and not normalized_must_keep_terms:
        return _build_default_fts_query_specs(
            content_tokens,
            original_text=message_text,
        )[:max_queries]

    content_tokens = _stable_token_union(content_tokens, normalized_must_keep_terms)
    if not content_tokens and not normalized_quoted_phrases:
        return []

    specs: list[RetrievalFtsQuerySpec] = []
    seen_queries: set[str] = set()
    for phrase in normalized_quoted_phrases:
        quoted_query = f'"{phrase}"'
        _append_fts_query_spec(
            specs,
            seen_queries,
            query=quoted_query,
            kind="quoted_phrase",
        )

    if content_tokens:
        anchor_tokens = _stable_token_union(
            normalized_must_keep_terms,
            _collect_tokens_from_values(normalized_quoted_phrases),
        )
        if anchor_tokens and normalized_must_keep_terms and not normalized_quoted_phrases:
            anchor_first_query = " ".join(
                _stable_token_union(anchor_tokens, content_tokens)[:_MAX_HINT_QUERY_TOKENS]
            )
            _append_fts_query_spec(
                specs,
                seen_queries,
                query=anchor_first_query,
                kind="anchor_first_and",
            )

        strong_query = " ".join(content_tokens[:_MAX_HINT_QUERY_TOKENS])
        _append_fts_query_spec(
            specs,
            seen_queries,
            query=strong_query,
            kind="sparse_and",
        )

        if exact_recall and len(normalized_must_keep_terms) >= 2:
            anchor_only_query = " ".join(
                normalized_must_keep_terms[:_MAX_HINT_QUERY_TOKENS]
            )
            _append_fts_query_spec(
                specs,
                seen_queries,
                query=anchor_only_query,
                kind="anchor_only_and",
            )

        if exact_recall and _should_add_must_keep_tail_query(
            content_tokens,
            normalized_must_keep_terms,
        ):
            tail_query = " ".join(
                normalized_must_keep_terms[1 : _MAX_HINT_QUERY_TOKENS + 1]
            )
            _append_fts_query_spec(
                specs,
                seen_queries,
                query=tail_query,
                kind="must_keep_tail_and",
            )

        if len(content_tokens) >= 3:
            broad_query = " OR ".join(content_tokens)
            _append_fts_query_spec(
                specs,
                seen_queries,
                query=broad_query,
                kind="broad_or",
            )

    return specs[:max_queries]


def _should_add_must_keep_tail_query(
    content_tokens: list[str],
    normalized_must_keep_terms: list[str],
) -> bool:
    return (
        len(normalized_must_keep_terms) >= 3
        and len(set(content_tokens) - set(normalized_must_keep_terms)) == 0
    )


def _append_fts_query_spec(
    specs: list[RetrievalFtsQuerySpec],
    seen_queries: set[str],
    *,
    query: str,
    kind: str,
) -> None:
    if query in seen_queries:
        return
    seen_queries.add(query)
    specs.append(RetrievalFtsQuerySpec(query=query, kind=kind))


def _build_default_fts_queries(
    content_tokens: list[str],
    *,
    original_text: str,
) -> list[str]:
    return [
        spec.query
        for spec in _build_default_fts_query_specs(
            content_tokens,
            original_text=original_text,
        )
    ]


def _build_default_fts_query_specs(
    content_tokens: list[str],
    *,
    original_text: str,
) -> list[RetrievalFtsQuerySpec]:
    if not content_tokens:
        return []

    specs: list[RetrievalFtsQuerySpec] = []
    if len(content_tokens) >= 4:
        specs.append(
            RetrievalFtsQuerySpec(
                query=" ".join(content_tokens[:4]),
                kind="default_and",
            )
        )
        specs.append(
            RetrievalFtsQuerySpec(
                query=" ".join(content_tokens[:3]),
                kind="default_short_and",
            )
        )
        specs.append(
            RetrievalFtsQuerySpec(
                query=" OR ".join(content_tokens),
                kind="broad_or",
            )
        )
    elif len(content_tokens) == 3:
        specs.append(
            RetrievalFtsQuerySpec(
                query=" ".join(content_tokens[:3]),
                kind="default_and",
            )
        )
        specs.append(
            RetrievalFtsQuerySpec(
                query=" OR ".join(content_tokens),
                kind="broad_or",
            )
        )
    else:
        specs.append(
            RetrievalFtsQuerySpec(
                query=" ".join(content_tokens),
                kind="default_and",
            )
        )
    return specs[:_MAX_RETRIEVAL_QUERIES]


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


def _append_anchor_alias_fts_query_specs(
    base_specs: list[RetrievalFtsQuerySpec],
    *,
    sparse_phrase: str,
    quoted_phrases: list[str],
    must_keep_terms: list[str],
    anchors: list[RuntimeAnchor],
    exact_recall: bool,
) -> list[RetrievalFtsQuerySpec]:
    """Materialize non-evidential runtime aliases as alternate FTS rewrites."""
    if not anchors:
        return base_specs

    specs = list(base_specs)
    seen_queries = {spec.query for spec in base_specs}
    alias_spec_count = 0
    for anchor in _anchors_with_aliases(anchors):
        for alias_surface in _alias_surfaces(anchor):
            variant = _anchor_alias_sparse_variant(
                sparse_phrase=sparse_phrase,
                quoted_phrases=quoted_phrases,
                must_keep_terms=must_keep_terms,
                anchor=anchor,
                alias_surface=alias_surface,
            )
            if variant is None:
                continue
            variant_sparse_phrase, variant_quoted_phrases, variant_must_keep_terms = (
                variant
            )
            for spec in build_retrieval_fts_query_specs(
                variant_sparse_phrase,
                quoted_phrases=variant_quoted_phrases,
                must_keep_terms=variant_must_keep_terms,
                exact_recall=exact_recall,
            ):
                if spec.query in seen_queries:
                    continue
                seen_queries.add(spec.query)
                specs.append(
                    RetrievalFtsQuerySpec(
                        query=spec.query,
                        kind=f"anchor_alias_{spec.kind}",
                    )
                )
                alias_spec_count += 1
                if alias_spec_count >= _MAX_ALIAS_QUERY_SPECS_PER_SUBQUERY:
                    return specs
    return specs


def _append_non_evidential_person_anchor_backoff_fts_query_specs(
    base_specs: list[RetrievalFtsQuerySpec],
    *,
    sparse_phrase: str,
    quoted_phrases: list[str],
    must_keep_terms: list[str],
    anchors: list[RuntimeAnchor],
    exact_recall: bool,
    query_type: str | None = None,
) -> list[RetrievalFtsQuerySpec]:
    """Add a narrow first-person evidence backoff for recall-oriented person queries."""
    if not (exact_recall or query_type == "broad_list"):
        return base_specs

    anchor_tokens = _non_evidential_person_anchor_tokens(anchors)
    if not anchor_tokens:
        return base_specs

    content_tokens = _tokens_without_anchor(
        _collect_tokens_from_values([sparse_phrase, *quoted_phrases, *must_keep_terms]),
        anchor_tokens,
    )
    prefix_tokens = [
        f"{token}*"
        for token in content_tokens
        if len(token) >= _MIN_PERSON_ANCHOR_BACKOFF_TOKEN_LENGTH
    ][:_MAX_PERSON_ANCHOR_BACKOFF_TOKENS]
    if not prefix_tokens:
        return base_specs

    seen_queries = {spec.query for spec in base_specs}
    specs: list[RetrievalFtsQuerySpec] = []

    def append_backoff_specs() -> None:
        _append_fts_query_spec(
            specs,
            seen_queries,
            query=" ".join(prefix_tokens),
            kind="non_evidential_person_anchor_backoff_prefix",
        )
        shortened_prefix_tokens = _shortened_prefix_backoff_tokens(content_tokens)
        if shortened_prefix_tokens and shortened_prefix_tokens != prefix_tokens:
            _append_fts_query_spec(
                specs,
                seen_queries,
                query=" OR ".join(shortened_prefix_tokens),
                kind="non_evidential_person_anchor_backoff_short_prefix_or",
            )

    inserted_backoff = False
    for spec in base_specs:
        if not inserted_backoff and spec.kind == "broad_or":
            append_backoff_specs()
            inserted_backoff = True
        specs.append(spec)
    if not inserted_backoff:
        append_backoff_specs()
    return specs


def _shortened_prefix_backoff_tokens(tokens: list[str]) -> list[str]:
    prefix_tokens: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if len(token) < _MIN_PERSON_ANCHOR_BACKOFF_TOKEN_LENGTH + 2:
            continue
        prefix = token[: max(_MIN_PERSON_ANCHOR_BACKOFF_TOKEN_LENGTH + 1, len(token) - 2)]
        if prefix == token:
            continue
        prefixed = f"{prefix}*"
        if prefixed in seen:
            continue
        seen.add(prefixed)
        prefix_tokens.append(prefixed)
        if len(prefix_tokens) >= _MAX_PERSON_ANCHOR_BACKOFF_TOKENS:
            break
    return prefix_tokens


def _non_evidential_person_anchor_tokens(anchors: list[RuntimeAnchor]) -> set[str]:
    tokens: set[str] = set()
    for anchor in anchors:
        if anchor.anchor_type != "person" or anchor.non_evidential is not True:
            continue
        tokens.update(_collect_tokens_from_values(_anchor_source_surfaces(anchor)))
    return tokens


def _tokens_without_anchor(tokens: list[str], anchor_tokens: set[str]) -> list[str]:
    if not anchor_tokens:
        return tokens
    return [token for token in tokens if token not in anchor_tokens]


def _anchors_with_aliases(anchors: list[RuntimeAnchor]) -> list[RuntimeAnchor]:
    selected: list[RuntimeAnchor] = []
    for anchor in anchors:
        if not anchor.aliases:
            continue
        selected.append(anchor)
        if len(selected) >= _MAX_ANCHORS_WITH_ALIAS_QUERIES:
            break
    return selected


def _alias_surfaces(anchor: RuntimeAnchor) -> list[str]:
    original_tokens = set(_collect_tokens_from_values([anchor.original_surface]))
    surfaces: list[str] = []
    seen: set[tuple[str, ...]] = set()
    for alias in anchor.aliases:
        surface = alias.surface.strip()
        if not surface:
            continue
        alias_tokens = tuple(_collect_tokens_from_values([surface]))
        if not alias_tokens or set(alias_tokens) == original_tokens:
            continue
        if alias_tokens in seen:
            continue
        seen.add(alias_tokens)
        surfaces.append(surface)
        if len(surfaces) >= _MAX_ALIASES_PER_ANCHOR:
            break
    return surfaces


def _anchor_alias_sparse_variant(
    *,
    sparse_phrase: str,
    quoted_phrases: list[str],
    must_keep_terms: list[str],
    anchor: RuntimeAnchor,
    alias_surface: str,
) -> tuple[str, list[str], list[str]] | None:
    source_surfaces = _anchor_source_surfaces(anchor)
    source_tokens = set(_collect_tokens_from_values(source_surfaces))
    if not source_tokens:
        return None

    variant_sparse_phrase = _replace_anchor_surface(
        sparse_phrase,
        source_surfaces=source_surfaces,
        alias_surface=alias_surface,
    )
    if variant_sparse_phrase is None:
        base_tokens = _collect_tokens_from_values([sparse_phrase])
        alias_tokens = _collect_tokens_from_values([alias_surface])
        remaining_tokens = [token for token in base_tokens if token not in source_tokens]
        variant_tokens = _stable_token_union(alias_tokens, remaining_tokens)[
            :_MAX_HINT_QUERY_TOKENS
        ]
        variant_sparse_phrase = " ".join(variant_tokens)
    if not variant_sparse_phrase.strip():
        return None

    variant_quoted_phrases = _replace_anchor_terms(
        quoted_phrases,
        source_surfaces=source_surfaces,
        source_tokens=source_tokens,
        alias_surface=alias_surface,
    )
    variant_must_keep_terms = _replace_anchor_terms(
        must_keep_terms,
        source_surfaces=source_surfaces,
        source_tokens=source_tokens,
        alias_surface=alias_surface,
    )
    if not _contains_token_set(variant_must_keep_terms, alias_surface):
        variant_must_keep_terms.insert(0, alias_surface)
    return variant_sparse_phrase, variant_quoted_phrases, variant_must_keep_terms


def _anchor_source_surfaces(anchor: RuntimeAnchor) -> list[str]:
    surfaces: list[str] = []
    seen: set[str] = set()
    for value in (anchor.original_surface, anchor.normalized_surface):
        surface = (value or "").strip()
        signature = surface.lower()
        if not surface or signature in seen:
            continue
        seen.add(signature)
        surfaces.append(surface)
    return surfaces


def _replace_anchor_terms(
    terms: list[str],
    *,
    source_surfaces: list[str],
    source_tokens: set[str],
    alias_surface: str,
) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for term in terms:
        replacement = _replace_anchor_surface(
            term,
            source_surfaces=source_surfaces,
            alias_surface=alias_surface,
        )
        if replacement is None and _term_only_mentions_anchor(term, source_tokens):
            replacement = alias_surface
        value = replacement or term
        value = value.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def _replace_anchor_surface(
    text: str,
    *,
    source_surfaces: list[str],
    alias_surface: str,
) -> str | None:
    for source_surface in source_surfaces:
        pattern = re.compile(
            rf"(?<!\w){re.escape(source_surface)}(?!\w)",
            re.IGNORECASE,
        )
        replaced, count = pattern.subn(alias_surface, text)
        if count:
            return replaced
    return None


def _term_only_mentions_anchor(term: str, source_tokens: set[str]) -> bool:
    term_tokens = set(_collect_tokens_from_values([term]))
    return bool(term_tokens) and term_tokens.issubset(source_tokens)


def _contains_token_set(terms: list[str], value: str) -> bool:
    value_tokens = set(_collect_tokens_from_values([value]))
    if not value_tokens:
        return False
    return any(set(_collect_tokens_from_values([term])) == value_tokens for term in terms)


class RetrievalPlanner:
    """Pure planner that converts query intelligence into a retrieval plan."""

    def build_plan(
        self,
        *,
        original_query: str,
        query_intelligence: QueryIntelligenceResult,
        conversation_context: ExtractionConversationContext,
        resolved_policy: ResolvedRetrievalPolicy,
        cold_start: bool,
    ) -> RetrievalPlan:
        base_scope_filter = self._ordered_scopes(
            resolved_policy.allowed_scopes,
            _DEFAULT_SCOPE_ORDER,
        )
        sub_query_plans = self._build_sub_query_plans(
            query_intelligence.sub_queries,
            query_intelligence.sparse_query_hints,
            query_intelligence.anchors,
            exact_recall=bool(query_intelligence.exact_recall_needed),
            query_type=query_intelligence.query_type,
        )
        no_searchable_tokens = not sub_query_plans
        plan = RetrievalPlan(
            original_query=original_query,
            assistant_mode_id=conversation_context.assistant_mode_id,
            workspace_id=conversation_context.workspace_id,
            conversation_id=conversation_context.conversation_id,
            user_persona_id=conversation_context.user_persona_id,
            platform_id=conversation_context.platform_id or "default",
            character_id=(
                conversation_context.character_id
                if conversation_context.character_id is not None
                else conversation_context.workspace_id
            ),
            active_presence_id=conversation_context.active_presence_id,
            active_space_id=conversation_context.active_space_id,
            active_space_boundary_mode=conversation_context.active_space_boundary_mode,
            active_mind_id=conversation_context.active_mind_id,
            mind_topology=conversation_context.mind_topology,
            active_embodiment_id=conversation_context.active_embodiment_id,
            cross_embodiment_mode=conversation_context.cross_embodiment_mode,
            active_realm_id=conversation_context.active_realm_id,
            cross_realm_mode=conversation_context.cross_realm_mode,
            incognito=conversation_context.incognito or conversation_context.isolated_mode,
            remember_across_chats=conversation_context.remember_across_chats,
            remember_across_devices=conversation_context.remember_across_devices,
            fts_queries=self._flatten_fts_queries(sub_query_plans),
            sub_query_plans=sub_query_plans,
            callback_bias=query_intelligence.callback_bias,
            raw_context_access_mode=query_intelligence.raw_context_access_mode,
            query_language=query_intelligence.query_language,
            answer_language=query_intelligence.answer_language,
            query_type=query_intelligence.query_type,
            scope_filter=base_scope_filter,
            status_filter=[MemoryStatus.ACTIVE],
            max_candidates=resolved_policy.retrieval_params.fts_limit,
            vector_limit=resolved_policy.retrieval_params.vector_limit,
            max_context_items=resolved_policy.retrieval_params.final_context_items,
            privacy_ceiling=resolved_policy.privacy_ceiling,
            allow_intimacy_context=resolved_policy.allow_intimacy_context,
            allow_private_sensitivity=resolved_policy.allow_private_sensitivity,
            retrieval_levels=list(query_intelligence.retrieval_levels),
            temporal_query_range=query_intelligence.temporal_range,
            consequence_search_enabled=False,
            require_evidence_regrounding=False,
            skip_retrieval=cold_start or no_searchable_tokens,
            exact_recall_mode=bool(query_intelligence.exact_recall_needed),
            exact_facets=list(query_intelligence.exact_facets),
        )
        if plan.exact_recall_mode and plan.query_type == "broad_list":
            plan.max_candidates = max(
                plan.max_candidates,
                _EXACT_BROAD_LIST_MIN_CANDIDATES,
            )

        sorted_needs = sorted(
            query_intelligence.needs,
            key=lambda need: (
                0 if need.need_type is NeedTrigger.MODE_SHIFT else 1,
                -need.confidence,
            ),
        )
        for need in sorted_needs:
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
                plan.allow_intimacy_context = resolved_policy.allow_intimacy_context
                plan.consequence_search_enabled = False
                plan.require_evidence_regrounding = False
            elif need.need_type is NeedTrigger.FRUSTRATION:
                plan.max_context_items = max(1, int(plan.max_context_items * 0.75))
            elif need.need_type is NeedTrigger.SENSITIVE_CONTEXT:
                # Sensitive-context detection should make retrieval more careful,
                # but the resolved policy remains the authority for what this
                # mode may see. Lowering the ceiling here hides the very facts a
                # same-user personal assistant may be allowed to recall.
                plan.max_candidates = self._increase_limit(plan.max_candidates)
                plan.vector_limit = self._increase_limit(plan.vector_limit)
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
        anchors: list[RuntimeAnchor],
        *,
        exact_recall: bool = False,
        query_type: str | None = None,
    ) -> list[PlannedSubQuery]:
        sparse_hints_by_sub_query = {
            hint.sub_query_text: hint for hint in sparse_query_hints
        }
        anchors_by_sub_query: dict[str, list[RuntimeAnchor]] = {}
        for anchor in anchors:
            anchors_by_sub_query.setdefault(anchor.sub_query_text, []).append(anchor)
        sub_query_plans: list[PlannedSubQuery] = []
        for sub_query in sub_queries:
            sparse_hint = sparse_hints_by_sub_query.get(sub_query)
            sparse_phrase = _resolve_sparse_phrase(sub_query, sparse_hint)
            quoted_phrases = list(sparse_hint.quoted_phrases) if sparse_hint is not None else []
            must_keep_terms = list(sparse_hint.must_keep_terms) if sparse_hint is not None else []
            fts_query_specs = build_retrieval_fts_query_specs(
                sparse_phrase,
                quoted_phrases=quoted_phrases,
                must_keep_terms=must_keep_terms,
                exact_recall=exact_recall,
            )
            fts_query_specs = _append_anchor_alias_fts_query_specs(
                fts_query_specs,
                sparse_phrase=sparse_phrase,
                quoted_phrases=quoted_phrases,
                must_keep_terms=must_keep_terms,
                anchors=anchors_by_sub_query.get(sub_query, []),
                exact_recall=exact_recall,
            )
            fts_query_specs = _append_non_evidential_person_anchor_backoff_fts_query_specs(
                fts_query_specs,
                sparse_phrase=sparse_phrase,
                quoted_phrases=quoted_phrases,
                must_keep_terms=must_keep_terms,
                anchors=anchors_by_sub_query.get(sub_query, []),
                exact_recall=exact_recall,
                query_type=query_type,
            )
            fts_queries = [spec.query for spec in fts_query_specs]
            if not fts_queries:
                continue
            sub_query_plans.append(
                PlannedSubQuery(
                    text=sub_query,
                    sparse_phrase=sparse_phrase,
                    quoted_phrases=quoted_phrases,
                    must_keep_terms=must_keep_terms,
                    fts_queries=fts_queries,
                    fts_query_kinds=[spec.kind for spec in fts_query_specs],
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
