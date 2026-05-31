"""Candidate search over SQLite and FTS indexes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import math
import re
from typing import Any

import aiosqlite

from atagia.core.clock import Clock
from atagia.core.config import Settings
from atagia.core.artifact_repository import ArtifactRepository
from atagia.core.memory_fact_facet_repository import fact_facet_value_key
from atagia.core.language_codes import (
    ISO_639_1_LANGUAGE_CODES,
    normalize_optional_iso_639_1_code,
)
from atagia.core.repositories import (
    MemoryObjectRepository,
    MessageRepository,
    _decode_json_columns,
    conversation_visibility_clause,
)
from atagia.core.verbatim_pin_repository import VerbatimPinRepository
from atagia.memory.embodiment_policy import (
    candidate_allows_embodiment_boundary,
    embodiment_visibility_sql_clause,
)
from atagia.memory.intimacy_boundary_policy import (
    candidate_allows_intimacy_boundary,
    coalesced_intimacy_sql_clause,
    memory_object_intimacy_sql_clause,
    strongest_intimacy_boundary,
)
from atagia.memory.mind_policy import (
    annotate_overseer_grants_for_rows,
    candidate_allows_mind_boundary,
    mind_visibility_sql_clause,
)
from atagia.memory.realm_policy import (
    annotate_realm_bridge_modes_for_rows,
    candidate_allows_realm_boundary,
    realm_visibility_sql_clause,
)
from atagia.memory.retrieval_planner import (
    RetrievalFtsQuerySpec,
    build_retrieval_fts_queries,
)
from atagia.memory.space_policy import (
    candidate_allows_space_boundary,
    space_visibility_sql_clause,
)
from atagia.models.schemas_memory import (
    MemoryCategory,
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    MemoryStatus,
    MindTopology,
    RetrievalPlan,
    RuntimeAliasGroupTrace,
    SpaceBoundaryMode,
    SummaryViewKind,
)
from atagia.services.embeddings import EmbeddingIndex, NoneBackend

_ALLOWED_CONSEQUENCE_MATCH_COLUMNS = frozenset(
    {
        "action_memory_id",
        "outcome_memory_id",
        "tendency_belief_id",
    }
)
# Channels participating in rank fusion. ``verbatim_evidence_search``
# carries FTS-backed conversation evidence and is weighted slightly
# lower than curated memory channels by default.
_CHANNEL_ORDER = (
    "verbatim_pin",
    "artifact_chunk",
    "fts",
    "fact_facet",
    "embedding",
    "consequence",
    "verbatim_evidence_search",
)
_QUERY_TYPE_CHANNEL_MULTIPLIERS: dict[str, dict[str, float]] = {
    "slot_fill": {
        "fact_facet": 1.15,
        "embedding": 0.75,
        "consequence": 0.85,
        "verbatim_evidence_search": 1.15,
    },
    "temporal": {
        "fact_facet": 1.20,
        "embedding": 0.75,
        "consequence": 0.85,
        "verbatim_evidence_search": 1.20,
    },
    "broad_list": {
        "fact_facet": 1.25,
        "verbatim_pin": 0.90,
        "artifact_chunk": 0.90,
        "fts": 0.90,
        "consequence": 0.90,
        "verbatim_evidence_search": 0.85,
    },
}
_EXACT_RECALL_CHANNEL_MULTIPLIERS: dict[str, float] = {
    "fact_facet": 1.25,
    "embedding": 0.65,
    "consequence": 0.85,
    "verbatim_evidence_search": 1.25,
}
_VERBATIM_EVIDENCE_WINDOW_ID_PREFIX = "vew_"
# Widest-to-narrowest order used to assign a synthetic scope to a
# verbatim evidence window. The widest scope compatible with both the
# plan and the window's owning conversation wins.
_EVIDENCE_SCOPE_BREADTH_ORDER = (
    MemoryScope.GLOBAL_USER,
    MemoryScope.WORKSPACE,
    MemoryScope.CONVERSATION,
    MemoryScope.EPHEMERAL_SESSION,
)
_SAFE_ALIAS_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CHANNEL_OVERFETCH_MULTIPLIER = 2
_TEMPORAL_OVERLAP_PRIORITY = 0
_TEMPORAL_UNKNOWN_PRIORITY = 1
_TEMPORAL_NON_OVERLAP_PRIORITY = 2
_SQL_LIKE_ESCAPE = "\\"
_SQL_LIKE_ESCAPE_CLAUSE = f"ESCAPE '{_SQL_LIKE_ESCAPE}'"
_STALE_EPHEMERAL_PRIORITY = 1
_SKIPPED_RAW_VERBATIM_MAX_CHARS = 12000
_SKIPPED_RAW_VERBATIM_SNIPPET_MAX_CHARS = 2000
_CORPUS_NEAR_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
_CORPUS_NEAR_MIN_TOKEN_LENGTH = 5
_CORPUS_NEAR_MAX_SOURCE_TOKENS = 6
_CORPUS_NEAR_MAX_TERMS = 6
_CORPUS_NEAR_MAX_TERMS_PER_TOKEN = 2
_CORPUS_NEAR_ROW_SCAN_LIMIT = 200
_RUNTIME_ALIAS_FTS_ALLOWED_ANCHOR_TYPES = frozenset({"concept", "attribute"})
_RUNTIME_ALIAS_FTS_MAX_TERMS = 8
_RUNTIME_ALIAS_FTS_KIND = "runtime_alias_or"
_RUNTIME_ALIAS_FTS_SOURCE = "alias_anchor"
_PERSISTED_SURFACE_FTS_SOURCE = "persisted_surface"
_PERSISTED_SURFACE_FTS_KIND_PREFIX = "persisted_surface_"
_SENSITIVITY_RANK = {
    "unknown": 0,
    "public": 1,
    "private": 2,
    "secret": 3,
}


def _canonical_scope_value(scope_value: str) -> str:
    try:
        scope = MemoryScope(scope_value)
    except ValueError:
        return scope_value
    canonical = MemoryObjectRepository.canonical_retrieval_scopes([scope])
    return canonical[0].value if canonical else scope_value


def _assert_safe_alias(alias: str) -> None:
    if not _SAFE_ALIAS_PATTERN.match(alias):
        raise ValueError(f"Unsafe SQL alias: {alias!r}")


def _sql_float_literal(value: float) -> str:
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"Unsafe BM25 column weight: {value!r}")
    return format(resolved, ".12g")


@dataclass(frozen=True, slots=True)
class _SourceMessageMetadataRows:
    """Batched memory metadata linked to a verbatim source-message window."""

    rows: list[dict[str, Any]]

    def pending_message_ids(self) -> set[str]:
        return {
            str(row["message_id"])
            for row in self.rows
            if str(row.get("status") or "")
            == MemoryStatus.PENDING_USER_CONFIRMATION.value
            and str(row.get("message_id") or "").strip()
        }

    def metadata_for_source_message_ids(
        self,
        source_message_ids: list[str],
    ) -> tuple[int, dict[str, Any], str, float]:
        source_id_set = {
            str(message_id).strip()
            for message_id in source_message_ids
            if str(message_id).strip()
        }
        rows = [
            row
            for row in self.rows
            if str(row.get("message_id") or "").strip() in source_id_set
        ]
        privacy_level = max(
            (int(row.get("privacy_level") or 0) for row in rows),
            default=0,
        )
        high_risk_metadata = self._high_risk_metadata(rows)
        boundary = strongest_intimacy_boundary(rows)
        confidences = [
            float(row.get("intimacy_boundary_confidence", 0.0) or 0.0)
            for row in rows
            if str(row.get("intimacy_boundary") or "ordinary") == boundary.value
        ]
        return (
            privacy_level,
            high_risk_metadata,
            boundary.value,
            max(confidences, default=0.0),
        )

    @staticmethod
    def _high_risk_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
        active_rows = [
            row
            for row in rows
            if str(row.get("status") or "") == MemoryStatus.ACTIVE.value
        ]
        if not active_rows:
            return {}
        if any(
            str(row.get("memory_category") or "")
            == MemoryCategory.PIN_OR_PASSWORD.value
            for row in active_rows
        ):
            return {
                "memory_category": MemoryCategory.PIN_OR_PASSWORD.value,
                "preserve_verbatim": True,
            }
        if any(bool(row.get("preserve_verbatim")) for row in active_rows):
            return {"preserve_verbatim": True}
        return {}


class CandidateSearch:
    """Executes retrieval plans against SQLite FTS indexes."""

    def __init__(
        self,
        connection: aiosqlite.Connection,
        clock: Clock,
        embedding_index: EmbeddingIndex | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._connection = connection
        self._clock = clock
        self._settings = settings or Settings.from_env()
        self._embedding_index = embedding_index or NoneBackend()
        self._rrf_k = self._settings.rrf_k
        self._message_repository = MessageRepository(connection, clock)
        self._artifact_repository = ArtifactRepository(connection, clock)
        self._verbatim_pin_repository = VerbatimPinRepository(connection, clock)
        self._verbatim_evidence_search_enabled = self._settings.verbatim_evidence_search_enabled
        self._verbatim_evidence_search_weight = float(
            self._settings.verbatim_evidence_search_rrf_weight
        )
        self._memory_bm25_sql = (
            "bm25(memory_objects_fts, "
            f"{_sql_float_literal(self._settings.memory_fts_canonical_bm25_weight)}, "
            f"{_sql_float_literal(self._settings.memory_fts_index_bm25_weight)})"
        )
        self._verbatim_evidence_search_limit = int(self._settings.verbatim_evidence_search_limit)
        self._verbatim_evidence_window_size = int(self._settings.verbatim_evidence_window_size)
        self._verbatim_evidence_window_overlap = int(self._settings.verbatim_evidence_window_overlap)
        self._fact_facet_retrieval_enabled = self._settings.fact_facet_retrieval_enabled
        self._fact_facet_structured_only = self._settings.fact_facet_structured_only
        self._fact_facet_retrieval_limit = int(self._settings.fact_facet_retrieval_limit)
        self._fact_facet_retrieval_weight = float(
            self._settings.fact_facet_retrieval_rrf_weight
        )

    async def search(
        self,
        plan: RetrievalPlan,
        user_id: str,
        *,
        fts_query_audit: list[dict[str, Any]] | None = None,
        runtime_alias_groups: list[RuntimeAliasGroupTrace] | None = None,
    ) -> list[dict[str, Any]]:
        if plan.skip_retrieval or plan.max_candidates <= 0:
            return []

        visibility_clauses, visibility_parameters = self._memory_visibility_clauses(
            plan,
            alias="mo",
        )
        if not visibility_clauses or not plan.status_filter or not plan.sub_query_plans:
            return []

        status_placeholders = ", ".join("?" for _ in plan.status_filter)
        privacy_filter, privacy_parameters = self._privacy_ceiling_filter_clause(
            plan,
            alias="mo",
        )
        intimacy_filter = self._memory_intimacy_filter_clause(plan, alias="mo")
        scope_order_case = self._scope_order_case(plan.scope_filter)
        temporal_order_case = self._temporal_order_case(plan, alias="mo")
        retrieval_level_clauses, retrieval_level_parameters = self._retrieval_level_clauses(plan)
        fts_limit = self._fts_channel_limit(plan)
        query = """
            SELECT
                mo.*,
                {memory_bm25_sql} AS rank
            FROM memory_objects_fts
            JOIN memory_objects AS mo ON mo._rowid = memory_objects_fts.rowid
            WHERE mo.user_id = ?
              AND {visibility_clauses}
              AND mo.status IN ({status_placeholders})
              AND {privacy_filter}
              AND {intimacy_filter}
              AND ({retrieval_level_clauses})
              AND mo.archived_by_conversation_id IS NULL
              AND {visibility_clause}
              AND memory_objects_fts MATCH ?
            ORDER BY {temporal_order_case}{scope_order_case}, rank ASC, mo.updated_at DESC
            LIMIT ?
        """.format(
            visibility_clauses=" AND ".join(visibility_clauses),
            status_placeholders=status_placeholders,
            privacy_filter=privacy_filter,
            retrieval_level_clauses=" OR ".join(retrieval_level_clauses),
            temporal_order_case=temporal_order_case,
            scope_order_case=scope_order_case,
            intimacy_filter=intimacy_filter,
            visibility_clause=conversation_visibility_clause("mo"),
            memory_bm25_sql=self._memory_bm25_sql,
        )
        surface_diagnostic_query = """
            SELECT
                mrs.id AS surface_id,
                mo.id AS memory_id
            FROM memory_retrieval_surfaces_fts
            JOIN memory_retrieval_surfaces AS mrs
              ON mrs._rowid = memory_retrieval_surfaces_fts.rowid
            JOIN memory_objects AS mo
              ON mo.id = mrs.memory_id
             AND mo.user_id = mrs.user_id
            WHERE mrs.user_id = ?
              AND mo.user_id = ?
              AND mrs.status = 'active'
              AND mrs.non_evidential = 1
              AND {visibility_clauses}
              AND mo.status IN ({status_placeholders})
              AND {privacy_filter}
              AND {intimacy_filter}
              AND ({retrieval_level_clauses})
              AND mo.archived_by_conversation_id IS NULL
              AND {visibility_clause}
              AND memory_retrieval_surfaces_fts MATCH ?
            ORDER BY mrs.updated_at DESC, mrs.id ASC
            LIMIT ?
        """.format(
            visibility_clauses=" AND ".join(visibility_clauses),
            status_placeholders=status_placeholders,
            privacy_filter=privacy_filter,
            retrieval_level_clauses=" OR ".join(retrieval_level_clauses),
            intimacy_filter=intimacy_filter,
            visibility_clause=conversation_visibility_clause("mo"),
        )
        surface_candidate_query = """
            SELECT
                mo.*,
                0.0 AS rank
            FROM memory_retrieval_surfaces_fts
            JOIN memory_retrieval_surfaces AS mrs
              ON mrs._rowid = memory_retrieval_surfaces_fts.rowid
            JOIN memory_objects AS mo
              ON mo.id = mrs.memory_id
             AND mo.user_id = mrs.user_id
            WHERE mrs.user_id = ?
              AND mo.user_id = ?
              AND mrs.status = 'active'
              AND mrs.non_evidential = 1
              AND mrs.preserve_verbatim = 0
              AND {visibility_clauses}
              AND mo.status IN ({status_placeholders})
              AND {privacy_filter}
              AND {intimacy_filter}
              AND ({retrieval_level_clauses})
              AND mo.archived_by_conversation_id IS NULL
              AND {visibility_clause}
              AND memory_retrieval_surfaces_fts MATCH ?
            ORDER BY {temporal_order_case}{scope_order_case}, mrs.updated_at DESC, mo.updated_at DESC, mo.id ASC
            LIMIT ?
        """.format(
            visibility_clauses=" AND ".join(visibility_clauses),
            status_placeholders=status_placeholders,
            privacy_filter=privacy_filter,
            retrieval_level_clauses=" OR ".join(retrieval_level_clauses),
            temporal_order_case=temporal_order_case,
            scope_order_case=scope_order_case,
            intimacy_filter=intimacy_filter,
            visibility_clause=conversation_visibility_clause("mo"),
        )

        aggregated: dict[str, dict[str, Any]] = {}
        for sub_query in plan.sub_query_plans:
            sub_query_aggregated: dict[str, dict[str, Any]] = {}
            sub_query_memory_fts_raw_rows = 0
            executed_fts_queries: set[str] = set()
            persisted_surface_search_enabled = (
                plan.exact_recall_mode or plan.query_type == "slot_fill"
            )
            for fts_query_index, fts_query in enumerate(sub_query.fts_queries):
                fts_query_kind = self._fts_query_kind_at(
                    sub_query,
                    fts_query_index,
                )
                fts_candidates, raw_rows = await self._search_memory_fts_query(
                    query,
                    user_id=user_id,
                    visibility_parameters=visibility_parameters,
                    status_values=[status.value for status in plan.status_filter],
                    privacy_parameters=privacy_parameters,
                    retrieval_level_parameters=retrieval_level_parameters,
                    plan=plan,
                    sub_query_text=str(sub_query.text),
                    fts_query=str(fts_query),
                    fts_query_kind=fts_query_kind,
                    fts_limit=fts_limit,
                    fts_query_audit=fts_query_audit,
                )
                sub_query_memory_fts_raw_rows += raw_rows
                executed_fts_queries.add(str(fts_query))
                if persisted_surface_search_enabled:
                    surface_candidates, surface_raw_rows = await self._search_persisted_surface_fts_query(
                        surface_candidate_query,
                        user_id=user_id,
                        visibility_parameters=visibility_parameters,
                        status_values=[status.value for status in plan.status_filter],
                        privacy_parameters=privacy_parameters,
                        retrieval_level_parameters=retrieval_level_parameters,
                        plan=plan,
                        sub_query_text=str(sub_query.text),
                        fts_query=str(fts_query),
                        fts_query_kind=fts_query_kind,
                        fts_limit=fts_limit,
                        fts_query_audit=fts_query_audit,
                    )
                    sub_query_memory_fts_raw_rows += surface_raw_rows
                    self._merge_channel_candidates(
                        sub_query_aggregated,
                        surface_candidates,
                        channel="fts",
                        plan=plan,
                    )
                else:
                    await self._audit_persisted_surface_fts_query(
                        surface_diagnostic_query,
                        user_id=user_id,
                        visibility_parameters=visibility_parameters,
                        status_values=[status.value for status in plan.status_filter],
                        privacy_parameters=privacy_parameters,
                        retrieval_level_parameters=retrieval_level_parameters,
                        plan=plan,
                        sub_query_text=str(sub_query.text),
                        fts_query=str(fts_query),
                        fts_query_kind=fts_query_kind,
                        fts_limit=fts_limit,
                        fts_query_audit=fts_query_audit,
                    )
                self._merge_channel_candidates(
                    sub_query_aggregated,
                    fts_candidates,
                    channel="fts",
                    plan=plan,
                )
            if plan.exact_recall_mode or plan.query_type == "slot_fill":
                for spec in self._build_runtime_alias_fts_query_specs(
                    sub_query=sub_query,
                    runtime_alias_groups=runtime_alias_groups,
                ):
                    if spec.query in executed_fts_queries:
                        continue
                    fts_candidates, raw_rows = await self._search_memory_fts_query(
                        query,
                        user_id=user_id,
                        visibility_parameters=visibility_parameters,
                        status_values=[status.value for status in plan.status_filter],
                        privacy_parameters=privacy_parameters,
                        retrieval_level_parameters=retrieval_level_parameters,
                        plan=plan,
                        sub_query_text=str(sub_query.text),
                        fts_query=spec.query,
                        fts_query_kind=spec.kind,
                        fts_limit=fts_limit,
                        fts_query_audit=fts_query_audit,
                        fts_query_source=_RUNTIME_ALIAS_FTS_SOURCE,
                        non_evidential=True,
                    )
                    sub_query_memory_fts_raw_rows += raw_rows
                    executed_fts_queries.add(spec.query)
                    self._merge_channel_candidates(
                        sub_query_aggregated,
                        fts_candidates,
                        channel="fts",
                        plan=plan,
                    )
            if plan.exact_recall_mode and sub_query_memory_fts_raw_rows == 0:
                corpus_near_specs = await self._build_corpus_near_fts_query_specs(
                    plan=plan,
                    user_id=user_id,
                    sub_query=sub_query,
                    visibility_parameters=visibility_parameters,
                    status_values=[status.value for status in plan.status_filter],
                    privacy_parameters=privacy_parameters,
                    retrieval_level_parameters=retrieval_level_parameters,
                )
                for spec in corpus_near_specs:
                    if spec.query in executed_fts_queries:
                        continue
                    fts_candidates, _raw_rows = await self._search_memory_fts_query(
                        query,
                        user_id=user_id,
                        visibility_parameters=visibility_parameters,
                        status_values=[status.value for status in plan.status_filter],
                        privacy_parameters=privacy_parameters,
                        retrieval_level_parameters=retrieval_level_parameters,
                        plan=plan,
                        sub_query_text=str(sub_query.text),
                        fts_query=spec.query,
                        fts_query_kind=spec.kind,
                        fts_limit=fts_limit,
                        fts_query_audit=fts_query_audit,
                    )
                    executed_fts_queries.add(spec.query)
                    self._merge_channel_candidates(
                        sub_query_aggregated,
                        fts_candidates,
                        channel="fts",
                        plan=plan,
                    )
            fact_facet_candidates = await self._search_fact_facets(
                plan=plan,
                user_id=user_id,
                sub_query=sub_query,
            )
            self._merge_channel_candidates(
                sub_query_aggregated,
                fact_facet_candidates,
                channel="fact_facet",
                plan=plan,
            )
            verbatim_pin_candidates = await self._search_verbatim_pins(
                plan=plan,
                user_id=user_id,
                sub_query=sub_query,
            )
            self._merge_channel_candidates(
                sub_query_aggregated,
                verbatim_pin_candidates,
                channel="verbatim_pin",
                plan=plan,
            )
            artifact_chunk_candidates = await self._search_artifact_chunks(
                plan=plan,
                user_id=user_id,
                sub_query=sub_query,
                linked_candidates=sub_query_aggregated.values(),
                include_lexical=True,
            )
            self._merge_channel_candidates(
                sub_query_aggregated,
                artifact_chunk_candidates,
                channel="artifact_chunk",
                plan=plan,
            )
            if self._needs_consequence_chain_search(plan):
                consequence_candidates = await self._search_consequence_chains(
                    plan=plan,
                    user_id=user_id,
                    fts_query=sub_query.fts_queries[0],
                )
                self._merge_channel_candidates(
                    sub_query_aggregated,
                    consequence_candidates,
                    channel="consequence",
                    plan=plan,
                )
            embedding_candidates = await self.search_by_embedding(
                query_text=sub_query.text,
                user_id=user_id,
                plan=plan,
                embedding_index=self._embedding_index,
            )
            self._merge_channel_candidates(
                sub_query_aggregated,
                embedding_candidates,
                channel="embedding",
                plan=plan,
            )
            # FTS-backed evidence lane. Each sub-query contributes
            # conversation windows from the messages table alongside
            # the memory-object channels. Privacy filtering runs at the
            # SQL layer, so results that reach here are already scoped
            # to the current ceiling.
            verbatim_evidence_search_candidates = await self._search_verbatim_evidence(
                plan=plan,
                user_id=user_id,
                sub_query=sub_query,
            )
            self._merge_channel_candidates(
                sub_query_aggregated,
                verbatim_evidence_search_candidates,
                channel="verbatim_evidence_search",
                plan=plan,
            )
            linked_artifact_chunk_candidates = await self._search_artifact_chunks(
                plan=plan,
                user_id=user_id,
                sub_query=sub_query,
                linked_candidates=sub_query_aggregated.values(),
                include_lexical=False,
            )
            self._merge_channel_candidates(
                sub_query_aggregated,
                linked_artifact_chunk_candidates,
                channel="artifact_chunk",
                plan=plan,
            )
            # Collapse evidence windows that overlap memory-object
            # candidates already retrieved for this sub-query. Memories
            # are more focused; verbatim evidence is the recall safety net.
            self._dedupe_verbatim_evidence_windows_against_memories(
                sub_query_aggregated,
                plan=plan,
            )
            if plan.callback_bias and sub_query_aggregated:
                await self._populate_callback_source_flags(
                    list(sub_query_aggregated.values()),
                    user_id=user_id,
                )
            ranked_sub_query_candidates = self._assign_position_ranks(
                self._sort_candidates(sub_query_aggregated.values(), plan)
            )
            self._merge_sub_query_candidates(
                aggregated,
                ranked_sub_query_candidates,
                plan=plan,
                sub_query_text=sub_query.text,
                total_sub_queries=len(plan.sub_query_plans),
            )

        # Cross-sub-query dedup pass: a verbatim evidence window fetched
        # under one sub-query may overlap a memory object fetched under a
        # different sub-query. The per-sub-query dedup call above only
        # sees one sub-query at a time, so repeat it against the fully
        # merged set before final ranking.
        self._dedupe_verbatim_evidence_windows_against_memories(
            aggregated,
            plan=plan,
        )
        if plan.callback_bias and aggregated:
            await self._populate_callback_source_flags(
                list(aggregated.values()),
                user_id=user_id,
            )
        return self._sort_candidates(aggregated.values(), plan)[: plan.max_candidates]

    async def _search_memory_fts_query(
        self,
        query: str,
        *,
        user_id: str,
        visibility_parameters: list[Any],
        status_values: list[str],
        privacy_parameters: list[Any],
        retrieval_level_parameters: list[Any],
        plan: RetrievalPlan,
        sub_query_text: str,
        fts_query: str,
        fts_query_kind: str,
        fts_limit: int,
        fts_query_audit: list[dict[str, Any]] | None,
        fts_query_source: str = "planned",
        non_evidential: bool = True,
    ) -> tuple[list[dict[str, Any]], int]:
        parameters = (
            user_id,
            *visibility_parameters,
            *status_values,
            *privacy_parameters,
            *retrieval_level_parameters,
            plan.conversation_id,
            fts_query,
            *self._temporal_order_parameters(plan),
            fts_limit,
        )
        cursor = await self._connection.execute(query, parameters)
        rows = await cursor.fetchall()
        raw_rows = len(rows)
        self._record_fts_query_audit(
            fts_query_audit,
            sub_query_text=sub_query_text,
            fts_query=fts_query,
            fts_query_kind=fts_query_kind,
            raw_rows=raw_rows,
            source=fts_query_source,
            non_evidential=non_evidential,
        )
        decoded_rows = [
            decoded
            for decoded in (_decode_json_columns(row) for row in rows)
            if decoded is not None
        ]
        await self._annotate_overseer_grants(decoded_rows, plan)
        await self._annotate_realm_bridge_modes(decoded_rows, plan)
        fts_candidates = self._assign_position_ranks(
            [
                self._annotate_fts_candidate(decoded)
                for decoded in decoded_rows
            ]
        )
        fts_candidates = [
            self._annotate_fts_query_match(
                candidate,
                sub_query_text=sub_query_text,
                fts_query=fts_query,
                fts_query_kind=fts_query_kind,
                fts_query_source=fts_query_source,
                non_evidential=non_evidential,
            )
            for candidate in fts_candidates
        ]
        return fts_candidates, raw_rows

    async def _search_persisted_surface_fts_query(
        self,
        query: str,
        *,
        user_id: str,
        visibility_parameters: list[Any],
        status_values: list[str],
        privacy_parameters: list[Any],
        retrieval_level_parameters: list[Any],
        plan: RetrievalPlan,
        sub_query_text: str,
        fts_query: str,
        fts_query_kind: str,
        fts_limit: int,
        fts_query_audit: list[dict[str, Any]] | None,
    ) -> tuple[list[dict[str, Any]], int]:
        persisted_kind = f"{_PERSISTED_SURFACE_FTS_KIND_PREFIX}{fts_query_kind}"
        parameters = (
            user_id,
            user_id,
            *visibility_parameters,
            *status_values,
            *privacy_parameters,
            *retrieval_level_parameters,
            plan.conversation_id,
            fts_query,
            *self._temporal_order_parameters(plan),
            fts_limit,
        )
        cursor = await self._connection.execute(query, parameters)
        rows = await cursor.fetchall()
        raw_rows = len(rows)
        self._record_fts_query_audit(
            fts_query_audit,
            sub_query_text=sub_query_text,
            fts_query=fts_query,
            fts_query_kind=persisted_kind,
            raw_rows=raw_rows,
            source=_PERSISTED_SURFACE_FTS_SOURCE,
            non_evidential=True,
        )
        decoded_rows = [
            decoded
            for decoded in (_decode_json_columns(row) for row in rows)
            if decoded is not None
        ]
        await self._annotate_overseer_grants(decoded_rows, plan)
        await self._annotate_realm_bridge_modes(decoded_rows, plan)
        candidates = self._assign_position_ranks(
            [
                self._annotate_fts_candidate(decoded)
                for decoded in decoded_rows
            ]
        )
        return [
            self._annotate_fts_query_match(
                candidate,
                sub_query_text=sub_query_text,
                fts_query=fts_query,
                fts_query_kind=persisted_kind,
                fts_query_source=_PERSISTED_SURFACE_FTS_SOURCE,
                non_evidential=True,
            )
            for candidate in candidates
        ], raw_rows

    async def _audit_persisted_surface_fts_query(
        self,
        query: str,
        *,
        user_id: str,
        visibility_parameters: list[Any],
        status_values: list[str],
        privacy_parameters: list[Any],
        retrieval_level_parameters: list[Any],
        plan: RetrievalPlan,
        sub_query_text: str,
        fts_query: str,
        fts_query_kind: str,
        fts_limit: int,
        fts_query_audit: list[dict[str, Any]] | None,
    ) -> None:
        if fts_query_audit is None:
            return
        parameters = (
            user_id,
            user_id,
            *visibility_parameters,
            *status_values,
            *privacy_parameters,
            *retrieval_level_parameters,
            plan.conversation_id,
            fts_query,
            fts_limit,
        )
        cursor = await self._connection.execute(query, parameters)
        rows = await cursor.fetchall()
        raw_rows = len(rows)
        if raw_rows <= 0:
            return
        self._record_fts_query_audit(
            fts_query_audit,
            sub_query_text=sub_query_text,
            fts_query=fts_query,
            fts_query_kind=f"{_PERSISTED_SURFACE_FTS_KIND_PREFIX}{fts_query_kind}",
            raw_rows=raw_rows,
            source=_PERSISTED_SURFACE_FTS_SOURCE,
            non_evidential=True,
        )

    async def _search_fact_facets(
        self,
        *,
        plan: RetrievalPlan,
        user_id: str,
        sub_query: Any,
    ) -> list[dict[str, Any]]:
        if not self._fact_facet_search_applies(plan):
            return []

        terms = self._fact_facet_query_terms(plan, sub_query)
        if not terms:
            return []

        visibility_clauses, visibility_parameters = self._memory_visibility_clauses(
            plan,
            alias="mo",
        )
        if not visibility_clauses or not plan.status_filter:
            return []
        status_placeholders = ", ".join("?" for _ in plan.status_filter)
        privacy_filter, privacy_parameters = self._privacy_ceiling_filter_clause(
            plan,
            alias="mo",
        )
        intimacy_filter = self._memory_intimacy_filter_clause(plan, alias="mo")
        retrieval_level_clauses, retrieval_level_parameters = self._retrieval_level_clauses(
            plan,
            alias="mo",
        )
        if not retrieval_level_clauses:
            return []

        match_clauses: list[str] = []
        match_parameters: list[str] = []
        for term in terms:
            like_value = self._fact_facet_like_pattern(term)
            # SQLite LOWER() is ASCII-oriented unless ICU is enabled. The normalized
            # facet key carries language-agnostic matching; raw fields remain
            # best-effort surface fallbacks.
            match_clauses.append(
                "("
                f"LOWER(mff.value_norm_key) LIKE ? {_SQL_LIKE_ESCAPE_CLAUSE} "
                f"OR LOWER(mff.value_text) LIKE ? {_SQL_LIKE_ESCAPE_CLAUSE} "
                f"OR LOWER(mff.facet_label) LIKE ? {_SQL_LIKE_ESCAPE_CLAUSE} "
                f"OR LOWER(mff.subject_surface) LIKE ? {_SQL_LIKE_ESCAPE_CLAUSE}"
                ")"
            )
            match_parameters.extend([like_value, like_value, like_value, like_value])
        current_state_clause = (
            "AND mff.current_state = 1"
            if plan.coverage_mode == "current_state"
            else ""
        )
        surface_class_clause = (
            "AND mff.surface_class = 'structured'"
            if self._fact_facet_structured_only
            else ""
        )
        limit = min(
            max(0, self._fact_facet_retrieval_limit),
            max(1, plan.max_candidates * _CHANNEL_OVERFETCH_MULTIPLIER),
        )
        query = """
            SELECT
                mo.*,
                mff.id AS fact_facet_id,
                mff.memory_id AS fact_facet_memory_id,
                mff.source_message_id AS fact_facet_source_message_id,
                mff.source_span_id AS fact_facet_source_span_id,
                mff.source_hash AS fact_facet_source_hash,
                mff.subject_surface AS fact_facet_subject_surface,
                mff.subject_cluster_id AS fact_facet_subject_cluster_id,
                mff.surface_class AS fact_facet_surface_class,
                mff.facet_label AS fact_facet_facet_label,
                mff.value_text AS fact_facet_value_text,
                mff.value_norm_key AS fact_facet_value_norm_key,
                mff.value_type AS fact_facet_value_type,
                mff.assertion_kind AS fact_facet_assertion_kind,
                mff.list_group_key AS fact_facet_list_group_key,
                mff.support_kind AS fact_facet_support_kind,
                mff.observed_at AS fact_facet_observed_at,
                mff.valid_from AS fact_facet_valid_from,
                mff.valid_to AS fact_facet_valid_to,
                mff.current_state AS fact_facet_current_state,
                mff.temporal_phrase AS fact_facet_temporal_phrase,
                mff.temporal_anchor_at AS fact_facet_temporal_anchor_at,
                mff.resolved_interval_start AS fact_facet_resolved_interval_start,
                mff.resolved_interval_end AS fact_facet_resolved_interval_end,
                mff.temporal_granularity AS fact_facet_temporal_granularity,
                mff.temporal_resolution_type AS fact_facet_temporal_resolution_type,
                mff.temporal_confidence AS fact_facet_temporal_confidence,
                mff.language_code AS fact_facet_language_code,
                mff.confidence AS fact_facet_confidence,
                mff.schema_version AS fact_facet_schema_version,
                mff.created_at AS fact_facet_created_at,
                span.quote_text AS fact_facet_source_quote,
                span.seq AS fact_facet_source_seq,
                span.occurred_at AS fact_facet_source_occurred_at,
                span.conversation_id AS fact_facet_source_conversation_id
            FROM memory_fact_facets AS mff
            JOIN memory_objects AS mo
              ON mo.id = mff.memory_id
             AND mo.user_id = mff.user_id
            JOIN memory_evidence_spans AS span
              ON span.id = mff.source_span_id
             AND span.user_id = mff.user_id
            WHERE mff.user_id = ?
              AND mo.user_id = ?
              AND {visibility_clauses}
              AND mo.status IN ({status_placeholders})
              AND {privacy_filter}
              AND {intimacy_filter}
              AND ({retrieval_level_clauses})
              AND mo.archived_by_conversation_id IS NULL
              AND {visibility_clause}
              {current_state_clause}
              {surface_class_clause}
              AND ({match_clauses})
            ORDER BY
                mff.current_state DESC,
                COALESCE(mff.observed_at, mff.created_at) DESC,
                mff.confidence DESC,
                mff.id ASC
            LIMIT ?
        """.format(
            visibility_clauses=" AND ".join(visibility_clauses),
            status_placeholders=status_placeholders,
            privacy_filter=privacy_filter,
            intimacy_filter=intimacy_filter,
            retrieval_level_clauses=" OR ".join(retrieval_level_clauses),
            visibility_clause=conversation_visibility_clause("mo"),
            current_state_clause=current_state_clause,
            surface_class_clause=surface_class_clause,
            match_clauses=" OR ".join(match_clauses),
        )
        parameters = (
            user_id,
            user_id,
            *visibility_parameters,
            *[status.value for status in plan.status_filter],
            *privacy_parameters,
            *retrieval_level_parameters,
            plan.conversation_id,
            *match_parameters,
            limit,
        )
        cursor = await self._connection.execute(query, parameters)
        rows = await cursor.fetchall()
        decoded_rows = [
            decoded
            for decoded in (_decode_json_columns(row) for row in rows)
            if decoded is not None
        ]
        await self._annotate_overseer_grants(decoded_rows, plan)
        await self._annotate_realm_bridge_modes(decoded_rows, plan)
        candidates = [
            self._build_fact_facet_candidate(
                row,
                plan=plan,
                sub_query_text=str(sub_query.text),
                terms=terms,
            )
            for row in decoded_rows
        ]
        candidates = [
            candidate for candidate in candidates if self._matches_plan_filters(candidate, plan)
        ]
        grouped = self._best_fact_facet_candidates_by_value(candidates, plan)
        ordered = sorted(
            grouped,
            key=lambda candidate: (
                self._temporal_priority(candidate, plan),
                -float(candidate.get("fact_facet_match_score") or 0.0),
                -float(candidate.get("fact_facet_confidence") or 0.0),
                -self._updated_at_sort_key(str(candidate.get("updated_at"))),
                str(candidate["id"]),
            ),
        )
        ranked = self._assign_position_ranks(ordered[:limit])
        for candidate in ranked:
            candidate["channel_ranks"] = {
                channel: (
                    int(candidate["position_rank"]) if channel == "fact_facet" else None
                )
                for channel in _CHANNEL_ORDER
            }
        return ranked

    def _fact_facet_search_applies(self, plan: RetrievalPlan) -> bool:
        if not self._fact_facet_retrieval_enabled:
            return False
        if self._fact_facet_retrieval_limit <= 0:
            return False
        if plan.source_precision != "required":
            return False
        if plan.answer_shape in {"list", "temporal"}:
            return True
        return plan.answer_shape == "single_fact" and (
            plan.coverage_mode == "current_state" or plan.exact_recall_mode
        )

    @classmethod
    def _fact_facet_query_terms(cls, plan: RetrievalPlan, sub_query: Any) -> list[str]:
        values: list[str] = []
        for field_name in ("sparse_phrase", "text"):
            value = getattr(sub_query, field_name, None)
            if value:
                values.append(str(value))
        values.extend(str(value) for value in getattr(sub_query, "quoted_phrases", []) or [])
        values.extend(str(value) for value in getattr(sub_query, "must_keep_terms", []) or [])
        values.extend(str(facet.value) for facet in plan.exact_facets)
        terms: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized_phrase = fact_facet_value_key(value)
            if len(normalized_phrase) >= 2 and normalized_phrase not in seen:
                seen.add(normalized_phrase)
                terms.append(normalized_phrase)
            for match in _CORPUS_NEAR_TOKEN_PATTERN.finditer(value):
                token = fact_facet_value_key(match.group(0))
                if len(token) < 2:
                    continue
                if token in seen:
                    continue
                seen.add(token)
                terms.append(token)
                if len(terms) >= 12:
                    return terms
            if len(terms) >= 12:
                return terms
        return terms[:12]

    @staticmethod
    def _fact_facet_like_pattern(term: str) -> str:
        escaped = (
            str(term)
            .replace(_SQL_LIKE_ESCAPE, _SQL_LIKE_ESCAPE * 2)
            .replace("%", f"{_SQL_LIKE_ESCAPE}%")
            .replace("_", f"{_SQL_LIKE_ESCAPE}_")
        )
        return f"%{escaped}%"

    @classmethod
    def _build_fact_facet_candidate(
        cls,
        row: dict[str, Any],
        *,
        plan: RetrievalPlan,
        sub_query_text: str,
        terms: list[str],
    ) -> dict[str, Any]:
        fact_id = str(row["fact_facet_id"])
        base_memory_id = str(row["fact_facet_memory_id"])
        value_text = str(row.get("fact_facet_value_text") or "")
        facet_label = str(row.get("fact_facet_facet_label") or "")
        subject_surface = str(row.get("fact_facet_subject_surface") or "")
        observed_at = str(
            row.get("fact_facet_observed_at")
            or row.get("fact_facet_source_occurred_at")
            or row.get("updated_at")
        )
        source_message_id = str(row["fact_facet_source_message_id"])
        source_span_id = str(row["fact_facet_source_span_id"])
        source_quote = str(row.get("fact_facet_source_quote") or "")
        surface_class = str(row.get("fact_facet_surface_class") or "generic")
        match_score = cls._fact_facet_match_score(row, terms)
        payload_json = row.get("payload_json") or {}
        if not isinstance(payload_json, dict):
            payload_json = {}
        fact_payload = {
            "fact_id": fact_id,
            "memory_id": base_memory_id,
            "source_span_id": source_span_id,
            "source_message_id": source_message_id,
            "source_hash": row.get("fact_facet_source_hash"),
            "subject_surface": subject_surface,
            "subject_cluster_id": row.get("fact_facet_subject_cluster_id"),
            "surface_class": surface_class,
            "facet_label": facet_label,
            "value_text": value_text,
            "value_norm_key": row.get("fact_facet_value_norm_key"),
            "value_type": row.get("fact_facet_value_type"),
            "assertion_kind": row.get("fact_facet_assertion_kind"),
            "list_group_key": row.get("fact_facet_list_group_key"),
            "support_kind": row.get("fact_facet_support_kind"),
            "observed_at": observed_at,
            "current_state": bool(row.get("fact_facet_current_state")),
            "temporal_phrase": row.get("fact_facet_temporal_phrase"),
            "temporal_anchor_at": row.get("fact_facet_temporal_anchor_at"),
            "resolved_interval_start": row.get("fact_facet_resolved_interval_start"),
            "resolved_interval_end": row.get("fact_facet_resolved_interval_end"),
            "temporal_granularity": row.get("fact_facet_temporal_granularity"),
            "temporal_resolution_type": row.get("fact_facet_temporal_resolution_type"),
            "temporal_confidence": row.get("fact_facet_temporal_confidence"),
            "language_code": row.get("fact_facet_language_code"),
            "schema_version": row.get("fact_facet_schema_version"),
            "match_score": match_score,
        }
        updated_payload = dict(payload_json)
        updated_payload.update(
            {
                "source_kind_variant": "fact_facet",
                "source_memory_ids": [base_memory_id],
                "source_message_ids": [source_message_id],
                "source_span_ids": [source_span_id],
                "fact_facet": fact_payload,
                "subject_surface": subject_surface,
                "surface_class": surface_class,
                "facet_label": facet_label,
                "value_text": value_text,
                "value_norm_key": row.get("fact_facet_value_norm_key"),
                "value_type": row.get("fact_facet_value_type"),
                "display_text": value_text,
                "surface": value_text,
                "sub_query_text": sub_query_text,
            }
        )
        candidate = dict(row)
        candidate.update(
            {
                "id": fact_id,
                "fact_facet_memory_id": base_memory_id,
                "object_type": row.get("object_type") or MemoryObjectType.EVIDENCE.value,
                "canonical_text": cls._format_fact_facet_text(
                    subject_surface=subject_surface,
                    facet_label=facet_label,
                    value_text=value_text,
                ),
                "payload_json": updated_payload,
                "source_kind": MemorySourceKind.EXTRACTED.value,
                "confidence": float(row.get("fact_facet_confidence") or row.get("confidence") or 0.5),
                "valid_from": row.get("fact_facet_valid_from") or row.get("valid_from"),
                "valid_to": row.get("fact_facet_valid_to") or row.get("valid_to"),
                "temporal_type": (
                    row.get("fact_facet_temporal_resolution_type")
                    or row.get("temporal_type")
                    or "unknown"
                ),
                "updated_at": observed_at,
                "rank": -match_score,
                "fts_rank": -match_score,
                "retrieval_level": 0,
                "is_fact_facet_candidate": True,
                "fact_facet_match_score": match_score,
                "fact_facet_confidence": float(row.get("fact_facet_confidence") or 0.5),
                "fact_facet_surface_class": surface_class,
                "evidence_packets": [
                    {
                        "id": f"fact_facet:{fact_id}",
                        "memory_id": fact_id,
                        "support_kind": row.get("fact_facet_support_kind") or "direct",
                        "speaker_relation_to_subject": "unknown",
                        "confidence": float(row.get("fact_facet_confidence") or 0.5),
                        "confidence_details": {"source": "memory_fact_facets"},
                        "rationale": None,
                        "spans": [
                            {
                                "id": source_span_id,
                                "memory_id": base_memory_id,
                                "conversation_id": row.get("fact_facet_source_conversation_id")
                                or row.get("conversation_id"),
                                "message_id": source_message_id,
                                "span_role": "source",
                                "quote_text": source_quote,
                                "seq": row.get("fact_facet_source_seq"),
                                "occurred_at": row.get("fact_facet_source_occurred_at")
                                or observed_at,
                            }
                        ],
                    }
                ],
            }
        )
        candidate["retrieval_sources"] = ["fact_facet"]
        candidate["channel_ranks"] = {
            channel: None
            for channel in _CHANNEL_ORDER
        }
        candidate["retrieval_source"] = "fact_facet"
        candidate["matched_sub_queries"] = [sub_query_text]
        return candidate

    @staticmethod
    def _format_fact_facet_text(
        *,
        subject_surface: str,
        facet_label: str,
        value_text: str,
    ) -> str:
        subject = subject_surface.strip()
        facet = facet_label.strip()
        value = value_text.strip()
        if subject and facet:
            return f"{subject} / {facet}: {value}"
        if facet:
            return f"{facet}: {value}"
        return value

    @classmethod
    def _fact_facet_match_score(cls, row: dict[str, Any], terms: list[str]) -> float:
        fields = {
            "value": fact_facet_value_key(str(row.get("fact_facet_value_text") or "")),
            "value_key": fact_facet_value_key(str(row.get("fact_facet_value_norm_key") or "")),
            "facet": fact_facet_value_key(str(row.get("fact_facet_facet_label") or "")),
            "subject": fact_facet_value_key(str(row.get("fact_facet_subject_surface") or "")),
            "canonical": fact_facet_value_key(str(row.get("canonical_text") or "")),
        }
        score = 0.0
        for term in terms:
            normalized = fact_facet_value_key(term)
            if not normalized:
                continue
            if normalized in fields["value_key"] or normalized in fields["value"]:
                score += 3.0
            elif normalized in fields["facet"]:
                score += 2.0
            elif normalized in fields["subject"] or normalized in fields["canonical"]:
                score += 1.0
        return score

    @classmethod
    def _best_fact_facet_candidates_by_value(
        cls,
        candidates: list[dict[str, Any]],
        plan: RetrievalPlan,
    ) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
        for candidate in candidates:
            payload = candidate.get("payload_json") or {}
            fact_payload = payload.get("fact_facet") if isinstance(payload, dict) else None
            if not isinstance(fact_payload, dict):
                continue
            group_key = (
                str(fact_payload.get("facet_label") or ""),
                str(fact_payload.get("value_norm_key") or candidate["id"]),
                str(fact_payload.get("subject_surface") or ""),
            )
            current = grouped.get(group_key)
            if current is None or cls._fact_facet_candidate_sort_key(
                candidate,
                plan,
            ) < cls._fact_facet_candidate_sort_key(current, plan):
                grouped[group_key] = candidate
        return list(grouped.values())

    @classmethod
    def _fact_facet_candidate_sort_key(
        cls,
        candidate: dict[str, Any],
        plan: RetrievalPlan,
    ) -> tuple[int, float, float, float, str]:
        current_state = 0 if candidate.get("fact_facet_current_state") else 1
        temporal_priority = 0
        if plan.coverage_mode == "chronology":
            temporal_priority = 0 if candidate.get("valid_from") or candidate.get("valid_to") else 1
        return (
            current_state,
            float(temporal_priority),
            -float(candidate.get("fact_facet_match_score") or 0.0),
            -float(candidate.get("fact_facet_confidence") or 0.0),
            str(candidate["id"]),
        )

    def _build_runtime_alias_fts_query_specs(
        self,
        *,
        sub_query: Any,
        runtime_alias_groups: list[RuntimeAliasGroupTrace] | None,
    ) -> list[RetrievalFtsQuerySpec]:
        if not runtime_alias_groups:
            return []
        sub_query_text = str(getattr(sub_query, "text", "") or "")
        alias_queries: list[str] = []
        seen_queries: set[str] = set()
        for group in runtime_alias_groups:
            if group.sub_query_text != sub_query_text:
                continue
            if not self._runtime_alias_group_is_eligible(group):
                continue
            for alias in group.aliases:
                if alias.non_evidential is not True:
                    continue
                surface_queries = build_retrieval_fts_queries(
                    alias.surface,
                    exact_recall=True,
                )
                if not surface_queries:
                    continue
                query = surface_queries[0]
                if query in seen_queries:
                    continue
                seen_queries.add(query)
                alias_queries.append(query)
                if len(alias_queries) >= _RUNTIME_ALIAS_FTS_MAX_TERMS:
                    break
            if len(alias_queries) >= _RUNTIME_ALIAS_FTS_MAX_TERMS:
                break
        if not alias_queries:
            return []
        return [
            RetrievalFtsQuerySpec(
                query=" OR ".join(alias_queries),
                kind=_RUNTIME_ALIAS_FTS_KIND,
            )
        ]

    @staticmethod
    def _runtime_alias_group_is_eligible(group: RuntimeAliasGroupTrace) -> bool:
        if group.anchor_non_evidential is not True:
            return False
        if group.preserve_verbatim:
            return False
        return str(group.anchor_type) in _RUNTIME_ALIAS_FTS_ALLOWED_ANCHOR_TYPES

    async def _build_corpus_near_fts_query_specs(
        self,
        *,
        plan: RetrievalPlan,
        user_id: str,
        sub_query: Any,
        visibility_parameters: list[Any],
        status_values: list[str],
        privacy_parameters: list[Any],
        retrieval_level_parameters: list[Any],
    ) -> list[RetrievalFtsQuerySpec]:
        source_tokens = self._corpus_near_source_tokens(sub_query)
        if not source_tokens:
            return []
        near_terms: list[str] = []
        seen_terms: set[str] = set()
        for source_token in source_tokens:
            for term in await self._near_corpus_fts_terms(
                plan=plan,
                user_id=user_id,
                source_token=source_token,
                visibility_parameters=visibility_parameters,
                status_values=status_values,
                privacy_parameters=privacy_parameters,
                retrieval_level_parameters=retrieval_level_parameters,
            ):
                if term in seen_terms:
                    continue
                seen_terms.add(term)
                near_terms.append(term)
                if len(near_terms) >= _CORPUS_NEAR_MAX_TERMS:
                    break
            if len(near_terms) >= _CORPUS_NEAR_MAX_TERMS:
                break
        if not near_terms:
            return []
        return [
            RetrievalFtsQuerySpec(
                query=" OR ".join(near_terms),
                kind="corpus_near_or",
            )
        ]

    @staticmethod
    def _corpus_near_source_tokens(sub_query: Any) -> list[str]:
        blocked_tokens = CandidateSearch._corpus_near_blocked_tokens(sub_query)
        values: list[str] = []
        sparse_phrase = getattr(sub_query, "sparse_phrase", None)
        if sparse_phrase:
            values.append(str(sparse_phrase))
        else:
            values.append(str(getattr(sub_query, "text", "") or ""))
        values.extend(str(value) for value in getattr(sub_query, "must_keep_terms", []) or [])
        values.extend(str(value) for value in getattr(sub_query, "quoted_phrases", []) or [])

        tokens: list[str] = []
        seen: set[str] = set()
        for value in values:
            for match in _CORPUS_NEAR_TOKEN_PATTERN.finditer(value):
                raw_token = match.group(0)
                token = raw_token.lower().strip("_")
                if token in blocked_tokens:
                    continue
                if not CandidateSearch._is_corpus_near_source_token(raw_token, token):
                    continue
                if token in seen:
                    continue
                seen.add(token)
                tokens.append(token)
                if len(tokens) >= _CORPUS_NEAR_MAX_SOURCE_TOKENS:
                    return tokens
        return tokens

    @staticmethod
    def _corpus_near_blocked_tokens(sub_query: Any) -> set[str]:
        explicit_lower_tokens = CandidateSearch._corpus_near_explicit_lower_tokens(
            sub_query,
        )
        text_value = str(getattr(sub_query, "text", "") or "")
        must_keep_values = [
            str(value)
            for value in getattr(sub_query, "must_keep_terms", []) or []
        ]
        quoted_values = [
            str(value)
            for value in getattr(sub_query, "quoted_phrases", []) or []
        ]

        blocked: set[str] = set()
        for match in _CORPUS_NEAR_TOKEN_PATTERN.finditer(text_value):
            raw_token = match.group(0)
            token = raw_token.lower().strip("_")
            if CandidateSearch._is_corpus_near_source_token(raw_token, token):
                continue
            if (
                token in explicit_lower_tokens
                and CandidateSearch._is_sentence_case_near_token(raw_token, token)
            ):
                continue
            blocked.add(token)
        for value in [*must_keep_values, *quoted_values]:
            for match in _CORPUS_NEAR_TOKEN_PATTERN.finditer(value):
                raw_token = match.group(0)
                token = raw_token.lower().strip("_")
                if not CandidateSearch._is_corpus_near_source_token(raw_token, token):
                    blocked.add(token)
        return blocked

    @staticmethod
    def _corpus_near_explicit_lower_tokens(sub_query: Any) -> set[str]:
        values: list[str] = []
        sparse_phrase = getattr(sub_query, "sparse_phrase", None)
        if sparse_phrase:
            values.append(str(sparse_phrase))
        values.extend(str(value) for value in getattr(sub_query, "must_keep_terms", []) or [])

        tokens: set[str] = set()
        for value in values:
            for match in _CORPUS_NEAR_TOKEN_PATTERN.finditer(value):
                raw_token = match.group(0)
                token = raw_token.lower().strip("_")
                if CandidateSearch._is_corpus_near_source_token(raw_token, token):
                    tokens.add(token)
        return tokens

    @staticmethod
    def _is_sentence_case_near_token(raw_token: str, token: str) -> bool:
        if len(token) < _CORPUS_NEAR_MIN_TOKEN_LENGTH:
            return False
        if any(character.isdigit() for character in token):
            return False
        if not token or token in {"and", "or", "not", "near"}:
            return False
        return raw_token == token[:1].upper() + token[1:]

    @staticmethod
    def _is_corpus_near_source_token(raw_token: str, token: str) -> bool:
        if len(token) < _CORPUS_NEAR_MIN_TOKEN_LENGTH:
            return False
        if token != raw_token:
            return False
        if any(character.isdigit() for character in token):
            return False
        if not any(character.isalpha() for character in token):
            return False
        if token in {"and", "or", "not", "near"}:
            return False
        return True

    async def _near_corpus_fts_terms(
        self,
        *,
        plan: RetrievalPlan,
        user_id: str,
        source_token: str,
        visibility_parameters: list[Any],
        status_values: list[str],
        privacy_parameters: list[Any],
        retrieval_level_parameters: list[Any],
    ) -> list[str]:
        distance_limit = self._corpus_near_distance_limit(source_token)
        if distance_limit <= 0:
            return []

        min_length = max(_CORPUS_NEAR_MIN_TOKEN_LENGTH, len(source_token) - distance_limit)
        max_length = len(source_token) + distance_limit
        visibility_clauses, _visibility_parameters = self._memory_visibility_clauses(
            plan,
            alias="mo",
        )
        if not visibility_clauses:
            return []
        status_placeholders = ", ".join("?" for _ in status_values)
        privacy_filter, _privacy_parameters = self._privacy_ceiling_filter_clause(
            plan,
            alias="mo",
        )
        intimacy_filter = self._memory_intimacy_filter_clause(plan, alias="mo")
        retrieval_level_clauses, _retrieval_level_parameters = self._retrieval_level_clauses(plan)
        query = """
            SELECT
                mo.canonical_text AS canonical_text,
                mo.index_text AS index_text
            FROM memory_objects AS mo
            WHERE mo.user_id = ?
              AND {visibility_clauses}
              AND mo.status IN ({status_placeholders})
              AND {privacy_filter}
              AND {intimacy_filter}
              AND ({retrieval_level_clauses})
              AND mo.archived_by_conversation_id IS NULL
              AND {visibility_clause}
              AND (
                  mo.canonical_text LIKE ?
                  OR COALESCE(mo.index_text, '') LIKE ?
              )
            ORDER BY mo.updated_at DESC
            LIMIT ?
        """.format(
            visibility_clauses=" AND ".join(visibility_clauses),
            status_placeholders=status_placeholders,
            privacy_filter=privacy_filter,
            retrieval_level_clauses=" OR ".join(retrieval_level_clauses),
            intimacy_filter=intimacy_filter,
            visibility_clause=conversation_visibility_clause("mo"),
        )
        cursor = await self._connection.execute(
            query,
            (
                user_id,
                *visibility_parameters,
                *status_values,
                *privacy_parameters,
                *retrieval_level_parameters,
                plan.conversation_id,
                f"%{source_token[0]}%",
                f"%{source_token[0]}%",
                _CORPUS_NEAR_ROW_SCAN_LIMIT,
            ),
        )
        rows = await cursor.fetchall()
        seen_terms: set[str] = set()
        terms: list[str] = []
        for row in rows:
            for value in (row["canonical_text"], row["index_text"]):
                for match in _CORPUS_NEAR_TOKEN_PATTERN.finditer(str(value or "")):
                    term = match.group(0).lower().strip("_")
                    if term in seen_terms:
                        continue
                    seen_terms.add(term)
                    if term == source_token:
                        continue
                    if len(term) < min_length or len(term) > max_length:
                        continue
                    if not term.startswith(source_token[0]):
                        continue
                    if self._bounded_edit_distance(source_token, term, distance_limit) > distance_limit:
                        continue
                    terms.append(term)
                    if len(terms) >= _CORPUS_NEAR_MAX_TERMS_PER_TOKEN:
                        return terms
        return terms

    @staticmethod
    def _corpus_near_distance_limit(source_token: str) -> int:
        if len(source_token) >= 5:
            return 1
        return 0

    @staticmethod
    def _bounded_edit_distance(left: str, right: str, limit: int) -> int:
        if abs(len(left) - len(right)) > limit:
            return limit + 1
        previous = list(range(len(right) + 1))
        for left_index, left_char in enumerate(left, start=1):
            current = [left_index]
            row_min = current[0]
            for right_index, right_char in enumerate(right, start=1):
                insertion = current[right_index - 1] + 1
                deletion = previous[right_index] + 1
                substitution = previous[right_index - 1] + (left_char != right_char)
                value = min(insertion, deletion, substitution)
                current.append(value)
                row_min = min(row_min, value)
            if row_min > limit:
                return limit + 1
            previous = current
        return previous[-1]

    async def aggregate_retrievable_content_language_mix(
        self,
        *,
        user_id: str,
        scope_filter: list[MemoryScope],
        assistant_mode_id: str,
        workspace_id: str | None,
        conversation_id: str,
        privacy_ceiling: int,
        allow_intimacy_context: bool = False,
        user_persona_id: str | None = None,
        platform_id: str = "default",
        character_id: str | None = None,
        active_space_id: str | None = None,
        active_space_boundary_mode: SpaceBoundaryMode | str | None = None,
        active_mind_id: str | None = None,
        mind_topology: MindTopology | str | None = MindTopology.UNIMIND,
        active_embodiment_id: str | None = None,
        active_realm_id: str | None = None,
        incognito: bool = False,
        remember_across_chats: bool = True,
        remember_across_devices: bool = True,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        plan = RetrievalPlan(
            assistant_mode_id=assistant_mode_id,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id if character_id is not None else workspace_id,
            active_space_id=active_space_id,
            active_space_boundary_mode=active_space_boundary_mode,
            active_mind_id=active_mind_id,
            mind_topology=mind_topology or MindTopology.UNIMIND,
            active_embodiment_id=active_embodiment_id,
            active_realm_id=active_realm_id,
            incognito=incognito,
            remember_across_chats=remember_across_chats,
            remember_across_devices=remember_across_devices,
            scope_filter=scope_filter,
            status_filter=[MemoryStatus.ACTIVE],
            max_candidates=max(1, limit),
            max_context_items=1,
            privacy_ceiling=privacy_ceiling,
            allow_intimacy_context=allow_intimacy_context,
        )
        visibility_clauses, visibility_parameters = self._memory_visibility_clauses(
            plan,
            alias="mo",
        )
        if not visibility_clauses:
            return []
        valid_code_placeholders = ", ".join("?" for _code in ISO_639_1_LANGUAGE_CODES)
        valid_language_codes = sorted(ISO_639_1_LANGUAGE_CODES)
        query = """
            SELECT
                lc.value AS language_code,
                COUNT(*) AS memory_count,
                MAX(mo.updated_at) AS last_seen_at
            FROM memory_objects AS mo
            JOIN json_each(
                CASE
                    WHEN json_valid(mo.language_codes_json) = 1
                     AND json_type(mo.language_codes_json) = 'array'
                    THEN mo.language_codes_json
                    ELSE '[]'
                END
            ) AS lc
            WHERE mo.user_id = ?
              AND {visibility_clauses}
              AND mo.status = ?
              AND mo.privacy_level <= ?
              AND {intimacy_filter}
              AND mo.archived_by_conversation_id IS NULL
              AND {visibility_clause}
              AND mo.language_codes_json IS NOT NULL
              AND lc.type = 'text'
              AND LOWER(lc.value) IN ({valid_codes})
            GROUP BY lc.value
        """.format(
            visibility_clauses=" AND ".join(visibility_clauses),
            intimacy_filter=memory_object_intimacy_sql_clause(
                "mo",
                allow_intimacy_context=allow_intimacy_context,
            ),
            visibility_clause=conversation_visibility_clause("mo"),
            valid_codes=valid_code_placeholders,
        )
        cursor = await self._connection.execute(
            query,
            (
                user_id,
                *visibility_parameters,
                MemoryStatus.ACTIVE.value,
                privacy_ceiling,
                conversation_id,
                *valid_language_codes,
            ),
        )
        known_rows = [dict(row) for row in await cursor.fetchall()]
        unknown_query = """
            SELECT
                'unknown' AS language_code,
                COUNT(*) AS memory_count,
                MAX(mo.updated_at) AS last_seen_at
            FROM memory_objects AS mo
            WHERE mo.user_id = ?
              AND {visibility_clauses}
              AND mo.status = ?
              AND mo.privacy_level <= ?
              AND {intimacy_filter}
              AND mo.archived_by_conversation_id IS NULL
              AND {visibility_clause}
              AND (
                    mo.language_codes_json IS NULL
                    OR TRIM(mo.language_codes_json) = ''
                    OR json_valid(mo.language_codes_json) = 0
                    OR (
                        json_valid(mo.language_codes_json) = 1
                        AND COALESCE(json_type(mo.language_codes_json), '') != 'array'
                    )
                    OR json_array_length(
                        CASE
                            WHEN json_valid(mo.language_codes_json) = 1
                             AND json_type(mo.language_codes_json) = 'array'
                            THEN mo.language_codes_json
                            ELSE '[]'
                        END
                    ) = 0
                    OR (
                        json_valid(mo.language_codes_json) = 1
                        AND json_array_length(
                            CASE
                                WHEN json_valid(mo.language_codes_json) = 1
                                 AND json_type(mo.language_codes_json) = 'array'
                                THEN mo.language_codes_json
                                ELSE '[]'
                            END
                        ) > 0
                        AND NOT EXISTS (
                            SELECT 1
                            FROM json_each(
                                CASE
                                    WHEN json_valid(mo.language_codes_json) = 1
                                     AND json_type(mo.language_codes_json) = 'array'
                                    THEN mo.language_codes_json
                                    ELSE '[]'
                                END
                            ) AS language_code
                            WHERE language_code.type = 'text'
                              AND LOWER(language_code.value) IN ({valid_codes})
                        )
                    )
                  )
        """.format(
            visibility_clauses=" AND ".join(visibility_clauses),
            intimacy_filter=memory_object_intimacy_sql_clause(
                "mo",
                allow_intimacy_context=allow_intimacy_context,
            ),
            visibility_clause=conversation_visibility_clause("mo"),
            valid_codes=valid_code_placeholders,
        )
        unknown_cursor = await self._connection.execute(
            unknown_query,
            (
                user_id,
                *visibility_parameters,
                MemoryStatus.ACTIVE.value,
                privacy_ceiling,
                conversation_id,
                *valid_language_codes,
            ),
        )
        unknown_row = await unknown_cursor.fetchone()
        if unknown_row is not None and int(unknown_row["memory_count"] or 0) > 0:
            known_rows.append(dict(unknown_row))
        return self._coalesce_language_profile_rows(known_rows)[:limit]

    @staticmethod
    def _coalesce_language_profile_rows(
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for row in rows:
            raw_language_code = str(row.get("language_code") or "").strip().lower()
            language_code = (
                "unknown"
                if raw_language_code == "unknown"
                else normalize_optional_iso_639_1_code(raw_language_code)
            )
            if language_code is None:
                continue
            memory_count = int(row.get("memory_count") or 0)
            if memory_count <= 0:
                continue
            last_seen_at = str(row.get("last_seen_at") or "")
            existing = merged.get(language_code)
            if existing is None:
                merged[language_code] = {
                    "language_code": language_code,
                    "memory_count": memory_count,
                    "last_seen_at": last_seen_at,
                }
                continue
            existing["memory_count"] = int(existing["memory_count"]) + memory_count
            if last_seen_at > str(existing.get("last_seen_at") or ""):
                existing["last_seen_at"] = last_seen_at

        ordered = sorted(merged.values(), key=lambda row: str(row["language_code"]))
        ordered = sorted(
            ordered,
            key=lambda row: str(row.get("last_seen_at") or ""),
            reverse=True,
        )
        return sorted(
            ordered,
            key=lambda row: int(row.get("memory_count") or 0),
            reverse=True,
        )

    async def search_by_embedding(
        self,
        *,
        query_text: str,
        user_id: str,
        plan: RetrievalPlan,
        embedding_index: EmbeddingIndex,
    ) -> list[dict[str, Any]]:
        vector_limit = self._effective_embedding_limit(plan, embedding_index)
        if vector_limit == 0:
            return []
        matches = await embedding_index.search(
            query_text,
            user_id,
            self._embedding_search_limit(vector_limit),
        )
        if not matches:
            return []

        candidates: list[dict[str, Any]] = []
        for match in matches:
            row = await self._load_memory_object(
                match.memory_id,
                user_id,
                conversation_id=plan.conversation_id,
            )
            if row is None:
                continue
            await self._annotate_overseer_grants([row], plan)
            await self._annotate_realm_bridge_modes([row], plan)
            if not self._matches_plan_filters(row, plan):
                continue
            row["similarity_score"] = match.score
            row["embedding_similarity_score"] = match.score
            row["embedding_distance"] = float(match.metadata.get("distance", 0.0))
            if match.position_rank is not None:
                row["embedding_position_rank"] = int(match.position_rank)
                row["position_rank"] = int(match.position_rank)
            candidates.append(row)
        candidates = self._sort_embedding_candidates(candidates, plan)
        if candidates and all(candidate.get("position_rank") is not None for candidate in candidates):
            return candidates[:vector_limit]
        return self._sort_embedding_candidates(self._assign_position_ranks(candidates), plan)[:vector_limit]

    async def _search_verbatim_pins(
        self,
        *,
        plan: RetrievalPlan,
        user_id: str,
        sub_query: Any,
    ) -> list[dict[str, Any]]:
        if not (plan.exact_recall_mode or plan.raw_context_access_mode == "verbatim"):
            return []
        if not sub_query.fts_queries:
            return []

        aggregated: dict[str, dict[str, Any]] = {}
        queries_to_try = list(sub_query.fts_queries)
        if plan.exact_recall_mode:
            queries_to_try = []
            for index, fts_query in enumerate(sub_query.fts_queries):
                queries_to_try.append(fts_query)
                if index == 0 and aggregated:
                    break

        for fts_query in queries_to_try:
            rows = await self._verbatim_pin_repository.search_active_verbatim_pins(
                user_id=user_id,
                query=fts_query,
                privacy_ceiling=self._effective_privacy_ceiling(plan),
                scope_filter=plan.scope_filter,
                assistant_mode_id=plan.assistant_mode_id,
                workspace_id=plan.workspace_id,
                conversation_id=plan.conversation_id,
                limit=self._fts_channel_limit(plan),
                allow_intimacy_context=self._effective_allow_intimacy_context(plan),
                as_of=self._clock.now().isoformat(),
                user_persona_id=plan.user_persona_id,
                platform_id=plan.platform_id,
                character_id=plan.character_id,
                incognito=plan.incognito,
                remember_across_chats=plan.remember_across_chats,
                remember_across_devices=plan.remember_across_devices,
                allow_private_sensitivity=self._effective_allow_private_sensitivity(plan),
                active_space_id=plan.active_space_id,
                active_space_boundary_mode=plan.active_space_boundary_mode,
                active_mind_id=plan.active_mind_id,
                mind_topology=plan.mind_topology,
                active_embodiment_id=plan.active_embodiment_id,
                active_realm_id=plan.active_realm_id,
            )
            await self._annotate_overseer_grants(rows, plan)
            await self._annotate_realm_bridge_modes(rows, plan)
            for row in rows:
                pin_candidate = self._build_verbatim_pin_candidate(
                    row,
                    sub_query_text=sub_query.text,
                )
                if not self._matches_plan_filters(pin_candidate, plan):
                    continue
                memory_id = str(pin_candidate["id"])
                existing = aggregated.get(memory_id)
                if existing is None or float(pin_candidate.get("rank", 0.0)) < float(
                    existing.get("rank", 0.0)
                ):
                    aggregated[memory_id] = pin_candidate
            if plan.exact_recall_mode and aggregated:
                break

        ordered = sorted(
            aggregated.values(),
            key=lambda candidate: (
                float(candidate.get("rank", 0.0)),
                -self._updated_at_sort_key(str(candidate["updated_at"])),
                str(candidate["id"]),
            ),
        )
        return self._assign_position_ranks(ordered[: self._fts_channel_limit(plan)])

    async def _search_artifact_chunks(
        self,
        *,
        plan: RetrievalPlan,
        user_id: str,
        sub_query: Any,
        linked_candidates: Any,
        include_lexical: bool,
    ) -> list[dict[str, Any]]:
        if not (plan.exact_recall_mode or plan.raw_context_access_mode == "artifact"):
            return []
        if include_lexical and not sub_query.fts_queries:
            return []

        aggregated: dict[str, dict[str, Any]] = {}
        linked_rows = await self._artifact_repository.list_artifact_chunks_for_messages(
            user_id=user_id,
            message_ids=self._source_message_ids_from_candidates(linked_candidates),
            privacy_ceiling=self._effective_privacy_ceiling(plan),
            # Artifact rows do not yet carry the full namespace metadata, so
            # Phase 7 only exposes chunks attached to the active chat.
            scope_filter=[MemoryScope.CONVERSATION],
            assistant_mode_id=plan.assistant_mode_id,
            workspace_id=plan.workspace_id,
            conversation_id=plan.conversation_id,
            limit=self._fts_channel_limit(plan),
            allow_intimacy_context=self._effective_allow_intimacy_context(plan),
            user_persona_id=plan.user_persona_id,
            platform_id=plan.platform_id,
            character_id=plan.character_id,
            incognito=plan.incognito,
            remember_across_chats=plan.remember_across_chats,
            remember_across_devices=plan.remember_across_devices,
            allow_private_sensitivity=self._effective_allow_private_sensitivity(plan),
            active_space_id=plan.active_space_id,
            active_space_boundary_mode=plan.active_space_boundary_mode,
            active_mind_id=plan.active_mind_id,
            mind_topology=plan.mind_topology,
            active_embodiment_id=plan.active_embodiment_id,
            active_realm_id=plan.active_realm_id,
        )
        linked_artifact_candidates = [
            {
                **self._build_artifact_chunk_candidate(
                    row,
                    plan=plan,
                    sub_query_text=sub_query.text,
                ),
                "artifact_linked_source_message": True,
            }
            for row in linked_rows
        ]
        await self._annotate_overseer_grants(linked_artifact_candidates, plan)
        await self._annotate_realm_bridge_modes(linked_artifact_candidates, plan)
        for artifact_candidate in linked_artifact_candidates:
            if artifact_candidate.get("scope") is None or not self._matches_plan_filters(
                artifact_candidate,
                plan,
            ):
                continue
            aggregated[str(artifact_candidate["id"])] = artifact_candidate

        queries_to_try = self._artifact_fts_queries(plan, sub_query) if include_lexical else []

        for fts_query in queries_to_try:
            rows = await self._artifact_repository.search_artifact_chunks(
                user_id=user_id,
                query=fts_query,
                privacy_ceiling=self._effective_privacy_ceiling(plan),
                scope_filter=[MemoryScope.CONVERSATION],
                assistant_mode_id=plan.assistant_mode_id,
                workspace_id=plan.workspace_id,
                conversation_id=plan.conversation_id,
                limit=self._fts_channel_limit(plan),
                allow_intimacy_context=self._effective_allow_intimacy_context(plan),
                user_persona_id=plan.user_persona_id,
                platform_id=plan.platform_id,
                character_id=plan.character_id,
                incognito=plan.incognito,
                remember_across_chats=plan.remember_across_chats,
                remember_across_devices=plan.remember_across_devices,
                allow_private_sensitivity=self._effective_allow_private_sensitivity(plan),
                active_space_id=plan.active_space_id,
                active_space_boundary_mode=plan.active_space_boundary_mode,
                active_mind_id=plan.active_mind_id,
                mind_topology=plan.mind_topology,
                active_embodiment_id=plan.active_embodiment_id,
                active_realm_id=plan.active_realm_id,
            )
            artifact_candidates = [
                self._build_artifact_chunk_candidate(
                    row,
                    plan=plan,
                    sub_query_text=sub_query.text,
                )
                for row in rows
            ]
            await self._annotate_overseer_grants(artifact_candidates, plan)
            await self._annotate_realm_bridge_modes(artifact_candidates, plan)
            for artifact_candidate in artifact_candidates:
                if artifact_candidate.get("scope") is None or not self._matches_plan_filters(
                    artifact_candidate,
                    plan,
                ):
                    continue
                memory_id = str(artifact_candidate["id"])
                existing = aggregated.get(memory_id)
                if existing is None or float(artifact_candidate.get("rank", 0.0)) < float(
                    existing.get("rank", 0.0)
                ):
                    aggregated[memory_id] = artifact_candidate

        ordered = sorted(
            aggregated.values(),
            key=lambda candidate: (
                float(candidate.get("rank", 0.0)),
                -self._updated_at_sort_key(str(candidate["updated_at"])),
                str(candidate["id"]),
            ),
        )
        return self._assign_position_ranks(ordered[: self._fts_channel_limit(plan)])

    @staticmethod
    def _artifact_fts_queries(plan: RetrievalPlan, sub_query: Any) -> list[str]:
        queries: list[str] = []
        seen: set[str] = set()
        for fts_query in sub_query.fts_queries:
            if fts_query in seen:
                continue
            seen.add(fts_query)
            queries.append(fts_query)
        if plan.original_query:
            try:
                original_query_rewrites = build_retrieval_fts_queries(plan.original_query)
            except ValueError:
                original_query_rewrites = []
            for fts_query in original_query_rewrites:
                if fts_query in seen:
                    continue
                seen.add(fts_query)
                queries.append(fts_query)
        return queries

    @classmethod
    def _source_message_ids_from_candidates(cls, candidates: Any) -> list[str]:
        message_ids: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            for message_id in cls._candidate_source_message_ids(candidate):
                if message_id in seen:
                    continue
                seen.add(message_id)
                message_ids.append(message_id)
        return message_ids

    @staticmethod
    def _candidate_source_message_ids(candidate: Any) -> list[str]:
        if not isinstance(candidate, dict):
            return []
        raw_ids = candidate.get("verbatim_evidence_window_message_ids")
        if not raw_ids:
            payload = candidate.get("payload_json") or {}
            if isinstance(payload, dict):
                raw_ids = payload.get("source_message_ids")
        if not raw_ids and candidate.get("message_id"):
            raw_ids = [candidate.get("message_id")]
        if not isinstance(raw_ids, list):
            return []
        return [
            str(message_id).strip()
            for message_id in raw_ids
            if str(message_id).strip()
        ]

    async def _search_consequence_chains(
        self,
        *,
        plan: RetrievalPlan,
        user_id: str,
        fts_query: str,
    ) -> list[dict[str, Any]]:
        aggregated: dict[str, dict[str, Any]] = {}
        for match_column in ("action_memory_id", "outcome_memory_id", "tendency_belief_id"):
            rows = await self._search_consequence_chain_matches(
                plan=plan,
                user_id=user_id,
                fts_query=fts_query,
                match_column=match_column,
            )
            ranked_rows = self._assign_position_ranks(rows)
            for row in ranked_rows:
                memory_id = str(row["id"])
                current = aggregated.get(memory_id)
                if current is None or self._is_lower_position_rank(row, current):
                    aggregated[memory_id] = row
        return sorted(
            aggregated.values(),
            key=lambda candidate: (
                self._temporal_priority(candidate, plan),
                self._retrieval_level_priority(candidate, plan),
                self._scope_priority(candidate["scope"], plan.scope_filter),
                int(candidate["position_rank"]),
                -self._updated_at_sort_key(candidate["updated_at"]),
                str(candidate["id"]),
            ),
        )[: self._fts_channel_limit(plan)]

    async def _search_consequence_chain_matches(
        self,
        *,
        plan: RetrievalPlan,
        user_id: str,
        fts_query: str,
        match_column: str,
    ) -> list[dict[str, Any]]:
        if match_column not in _ALLOWED_CONSEQUENCE_MATCH_COLUMNS:
            raise ValueError(f"Invalid match column: {match_column}")
        candidate_scope_clauses, candidate_scope_parameters = self._coalesced_scope_clauses(
            plan,
            primary_alias="tendency",
            fallback_alias="outcome",
        )
        if not candidate_scope_clauses:
            return []
        action_visibility_clauses, action_visibility_parameters = self._consequence_source_visibility_clauses(
            plan,
            alias="action",
        )
        outcome_visibility_clauses, outcome_visibility_parameters = self._consequence_source_visibility_clauses(
            plan,
            alias="outcome",
        )
        tendency_visibility_clauses, tendency_visibility_parameters = self._memory_visibility_clauses(
            plan,
            alias="tendency",
        )
        if not action_visibility_clauses or not outcome_visibility_clauses or not tendency_visibility_clauses:
            return []
        chain_visibility_clauses, chain_visibility_parameters = self._memory_visibility_clauses(
            plan,
            alias="cc",
            include_space=False,
            include_mind=False,
            include_embodiment=False,
            include_realm=False,
        )
        if not chain_visibility_clauses:
            return []
        chain_relevance_clauses = ["cc.conversation_id = ?"]
        chain_relevance_parameters: list[Any] = [plan.conversation_id]
        if plan.workspace_id is not None:
            chain_relevance_clauses.append("cc.workspace_id = ?")
            chain_relevance_parameters.append(plan.workspace_id)
        if plan.character_id is not None:
            chain_relevance_clauses.append("cc.character_id = ?")
            chain_relevance_parameters.append(plan.character_id)
        status_placeholders = ", ".join("?" for _ in plan.status_filter)
        status_values = tuple(status.value for status in plan.status_filter)
        action_privacy_filter, action_privacy_parameters = self._privacy_ceiling_filter_clause(
            plan,
            alias="action",
        )
        outcome_privacy_filter, outcome_privacy_parameters = self._privacy_ceiling_filter_clause(
            plan,
            alias="outcome",
        )
        tendency_privacy_filter, tendency_privacy_parameters = self._privacy_ceiling_filter_clause(
            plan,
            alias="tendency",
        )
        candidate_intimacy_filter = self._coalesced_intimacy_filter_clause(
            plan,
            primary_alias="tendency",
            fallback_alias="outcome",
        )
        action_intimacy_filter = self._memory_intimacy_filter_clause(plan, alias="action")
        outcome_intimacy_filter = self._memory_intimacy_filter_clause(plan, alias="outcome")
        tendency_intimacy_filter = self._memory_intimacy_filter_clause(plan, alias="tendency")
        scope_order_case = self._coalesced_scope_order_case(
            plan.scope_filter,
            primary_alias="tendency",
            fallback_alias="outcome",
        )
        temporal_order_case = self._coalesced_temporal_order_case(
            plan,
            primary_alias="tendency",
            fallback_alias="outcome",
        )
        query = """
            SELECT
                COALESCE(tendency._rowid, outcome._rowid) AS _rowid,
                COALESCE(tendency.id, outcome.id) AS id,
                COALESCE(tendency.user_id, outcome.user_id) AS user_id,
                COALESCE(tendency.workspace_id, outcome.workspace_id) AS workspace_id,
                COALESCE(tendency.conversation_id, outcome.conversation_id) AS conversation_id,
                COALESCE(tendency.assistant_mode_id, outcome.assistant_mode_id) AS assistant_mode_id,
                COALESCE(tendency.user_persona_id, outcome.user_persona_id) AS user_persona_id,
                COALESCE(tendency.platform_id, outcome.platform_id) AS platform_id,
                COALESCE(tendency.character_id, outcome.character_id) AS character_id,
                COALESCE(tendency.space_id, outcome.space_id) AS space_id,
                COALESCE(tendency.space_boundary_mode, outcome.space_boundary_mode) AS space_boundary_mode,
                COALESCE(tendency.memory_owner_id, outcome.memory_owner_id) AS memory_owner_id,
                COALESCE(tendency.source_mind_id, outcome.source_mind_id) AS source_mind_id,
                COALESCE(tendency.embodiment_id, outcome.embodiment_id) AS embodiment_id,
                COALESCE(tendency.realm_id, outcome.realm_id) AS realm_id,
                COALESCE(tendency.object_type, outcome.object_type) AS object_type,
                COALESCE(tendency.scope, outcome.scope) AS scope,
                COALESCE(tendency.scope_canonical, outcome.scope_canonical) AS scope_canonical,
                COALESCE(tendency.canonical_text, outcome.canonical_text) AS canonical_text,
                COALESCE(tendency.extraction_hash, outcome.extraction_hash) AS extraction_hash,
                COALESCE(tendency.payload_json, outcome.payload_json) AS payload_json,
                COALESCE(tendency.source_kind, outcome.source_kind) AS source_kind,
                COALESCE(tendency.confidence, outcome.confidence) AS confidence,
                COALESCE(tendency.stability, outcome.stability) AS stability,
                COALESCE(tendency.vitality, outcome.vitality) AS vitality,
                COALESCE(tendency.maya_score, outcome.maya_score) AS maya_score,
                COALESCE(tendency.privacy_level, outcome.privacy_level) AS privacy_level,
                COALESCE(tendency.sensitivity, outcome.sensitivity) AS sensitivity,
                COALESCE(tendency.themes_json, outcome.themes_json) AS themes_json,
                COALESCE(tendency.platform_locked, outcome.platform_locked) AS platform_locked,
                COALESCE(tendency.platform_id_lock, outcome.platform_id_lock) AS platform_id_lock,
                COALESCE(tendency.temporal_type, outcome.temporal_type) AS temporal_type,
                COALESCE(tendency.valid_from, outcome.valid_from) AS valid_from,
                COALESCE(tendency.valid_to, outcome.valid_to) AS valid_to,
                COALESCE(tendency.status, outcome.status) AS status,
                COALESCE(tendency.created_at, outcome.created_at) AS created_at,
                COALESCE(tendency.updated_at, outcome.updated_at) AS updated_at,
                cc.id AS consequence_chain_id,
                action.canonical_text AS action_canonical_text,
                outcome.canonical_text AS outcome_canonical_text,
                tendency.canonical_text AS tendency_canonical_text,
                {memory_bm25_sql} AS rank
            FROM memory_objects_fts
            JOIN memory_objects AS matched ON matched._rowid = memory_objects_fts.rowid
            JOIN consequence_chains AS cc ON matched.id = cc.{match_column}
            LEFT JOIN memory_objects AS tendency ON tendency.id = cc.tendency_belief_id
            JOIN memory_objects AS action ON action.id = cc.action_memory_id
            JOIN memory_objects AS outcome ON outcome.id = cc.outcome_memory_id
            WHERE cc.user_id = ?
              AND cc.status = 'active'
              AND {chain_visibility_clauses}
              AND matched.user_id = ?
              AND action.user_id = ?
              AND action.status IN ({status_placeholders})
              AND {action_privacy_filter}
              AND action.archived_by_conversation_id IS NULL
              AND {action_visibility_clause}
              AND {action_visibility_clauses}
              AND {action_intimacy_filter}
              AND outcome.user_id = ?
              AND outcome.status IN ({status_placeholders})
              AND {outcome_privacy_filter}
              AND outcome.archived_by_conversation_id IS NULL
              AND {outcome_visibility_clause}
              AND {outcome_visibility_clauses}
              AND {outcome_intimacy_filter}
              AND (
                  tendency.id IS NULL
                  OR (
                      tendency.user_id = ?
                      AND tendency.status IN ({status_placeholders})
                      AND {tendency_privacy_filter}
                      AND tendency.archived_by_conversation_id IS NULL
                      AND {tendency_visibility_clause}
                      AND {tendency_visibility_clauses}
                      AND {tendency_intimacy_filter}
                  )
              )
              AND {candidate_intimacy_filter}
              AND ({candidate_scope_clauses})
              AND ({chain_relevance_clauses})
              AND memory_objects_fts MATCH ?
            ORDER BY
                {temporal_order_case}{scope_order_case},
                rank ASC,
                cc.confidence DESC,
                COALESCE(tendency.updated_at, outcome.updated_at) DESC
            LIMIT ?
        """.format(
            match_column=match_column,
            status_placeholders=status_placeholders,
            action_visibility_clause=conversation_visibility_clause("action"),
            action_visibility_clauses=" AND ".join(action_visibility_clauses),
            action_privacy_filter=action_privacy_filter,
            outcome_visibility_clause=conversation_visibility_clause("outcome"),
            outcome_visibility_clauses=" AND ".join(outcome_visibility_clauses),
            outcome_privacy_filter=outcome_privacy_filter,
            tendency_visibility_clause=conversation_visibility_clause("tendency"),
            tendency_visibility_clauses=" AND ".join(tendency_visibility_clauses),
            tendency_privacy_filter=tendency_privacy_filter,
            candidate_intimacy_filter=candidate_intimacy_filter,
            action_intimacy_filter=action_intimacy_filter,
            outcome_intimacy_filter=outcome_intimacy_filter,
            tendency_intimacy_filter=tendency_intimacy_filter,
            candidate_scope_clauses=" OR ".join(candidate_scope_clauses),
            chain_visibility_clauses=" AND ".join(chain_visibility_clauses),
            chain_relevance_clauses=" OR ".join(chain_relevance_clauses),
            temporal_order_case=temporal_order_case,
            scope_order_case=scope_order_case,
            memory_bm25_sql=self._memory_bm25_sql,
        )
        parameters = (
            user_id,
            *chain_visibility_parameters,
            user_id,
            user_id,
            *status_values,
            *action_privacy_parameters,
            plan.conversation_id,
            *action_visibility_parameters,
            user_id,
            *status_values,
            *outcome_privacy_parameters,
            plan.conversation_id,
            *outcome_visibility_parameters,
            user_id,
            *status_values,
            *tendency_privacy_parameters,
            plan.conversation_id,
            *tendency_visibility_parameters,
            *candidate_scope_parameters,
            *chain_relevance_parameters,
            fts_query,
            *self._temporal_order_parameters(plan),
            self._fts_channel_limit(plan),
        )
        cursor = await self._connection.execute(query, parameters)
        rows = await cursor.fetchall()
        decoded_rows = [
            decoded
            for decoded in (_decode_json_columns(row) for row in rows)
            if decoded is not None
        ]
        await self._annotate_overseer_grants(decoded_rows, plan)
        await self._annotate_realm_bridge_modes(decoded_rows, plan)
        return [
            decoded
            for decoded in decoded_rows
            if self._matches_plan_filters(decoded, plan)
        ]

    @staticmethod
    def _build_verbatim_pin_candidate(
        pin_row: dict[str, Any],
        *,
        sub_query_text: str,
    ) -> dict[str, Any]:
        payload = pin_row.get("payload_json") or {}
        if not isinstance(payload, dict):
            payload = {}
        rank = float(pin_row.get("rank") or 0.0)
        candidate_payload = {
            **payload,
            "verbatim_pin_id": str(pin_row["id"]),
            "verbatim_pin_target_kind": str(pin_row["target_kind"]),
            "verbatim_pin_target_id": str(pin_row["target_id"]),
            "intimacy_boundary": str(pin_row.get("intimacy_boundary") or "ordinary"),
            "intimacy_boundary_confidence": float(pin_row.get("intimacy_boundary_confidence") or 0.0),
        }
        if pin_row.get("space_id") is not None:
            candidate_payload["space_boundary"] = {
                "active_space_id": pin_row.get("space_id"),
                "boundary_mode": pin_row.get("space_boundary_mode"),
            }
        if pin_row.get("embodiment_id") is not None:
            candidate_payload["embodiment"] = {
                "active_embodiment_id": pin_row.get("embodiment_id"),
            }
        if pin_row.get("realm_id") is not None:
            candidate_payload["realm"] = {
                "active_realm_id": pin_row.get("realm_id"),
            }
            if pin_row.get("realm_bridge_mode") not in {None, "same"}:
                candidate_payload["realm"]["cross_realm_mode"] = pin_row.get(
                    "realm_bridge_mode"
                )
        candidate = {
            "_rowid": pin_row.get("_rowid"),
            "id": str(pin_row["id"]),
            "user_id": str(pin_row["user_id"]),
            "workspace_id": pin_row.get("workspace_id"),
            "conversation_id": pin_row.get("conversation_id"),
            "assistant_mode_id": pin_row.get("assistant_mode_id"),
            "user_persona_id": pin_row.get("user_persona_id"),
            "platform_id": pin_row.get("platform_id"),
            "character_id": pin_row.get("character_id"),
            "scope": str(pin_row["scope"]),
            "scope_canonical": str(pin_row.get("scope_canonical") or ""),
            "target_kind": str(pin_row["target_kind"]),
            "target_id": str(pin_row["target_id"]),
            "target_span_start": pin_row.get("target_span_start"),
            "target_span_end": pin_row.get("target_span_end"),
            "canonical_text": str(pin_row["canonical_text"]),
            "index_text": str(pin_row["index_text"]),
            "privacy_level": int(pin_row.get("privacy_level", 0)),
            "intimacy_boundary": str(pin_row.get("intimacy_boundary") or "ordinary"),
            "intimacy_boundary_confidence": float(pin_row.get("intimacy_boundary_confidence") or 0.0),
            "status": str(pin_row["status"]),
            "sensitivity": str(pin_row.get("sensitivity") or "unknown"),
            "themes_json": pin_row.get("themes_json") or [],
            "platform_locked": int(pin_row.get("platform_locked") or 0),
            "platform_id_lock": pin_row.get("platform_id_lock"),
            "space_id": pin_row.get("space_id"),
            "space_boundary_mode": pin_row.get("space_boundary_mode"),
            "memory_owner_id": pin_row.get("memory_owner_id"),
            "source_mind_id": pin_row.get("source_mind_id"),
            "embodiment_id": pin_row.get("embodiment_id"),
            "realm_id": pin_row.get("realm_id"),
            "realm_bridge_mode": pin_row.get("realm_bridge_mode"),
            "realm_relation": pin_row.get("realm_relation"),
            "reason": pin_row.get("reason"),
            "created_by": str(pin_row["created_by"]),
            "created_at": str(pin_row["created_at"]),
            "updated_at": str(pin_row["updated_at"]),
            "expires_at": pin_row.get("expires_at"),
            "deleted_at": pin_row.get("deleted_at"),
            "payload_json": candidate_payload,
            "source_kind": MemorySourceKind.VERBATIM.value,
            "object_type": MemoryObjectType.EVIDENCE.value,
            "confidence": 1.0,
            "stability": 1.0,
            "vitality": 0.0,
            "maya_score": 0.0,
            "temporal_type": "unknown",
            "valid_from": None,
            "valid_to": None,
            "rank": rank,
            "fts_rank": rank,
            "retrieval_sources": ["verbatim_pin"],
            "channel_ranks": {
                "verbatim_pin": None,
                "artifact_chunk": None,
                "fts": None,
                "fact_facet": None,
                "embedding": None,
                "consequence": None,
                "verbatim_evidence_search": None,
            },
            "matched_sub_queries": [sub_query_text],
            "is_verbatim_pin": True,
            "retrieval_level": 0,
        }
        candidate["channel_ranks"]["verbatim_pin"] = 1
        return candidate

    @staticmethod
    def _memory_visibility_clauses(
        plan: RetrievalPlan,
        *,
        alias: str = "mo",
        include_space: bool = True,
        include_mind: bool = True,
        include_embodiment: bool = True,
        include_realm: bool = True,
    ) -> tuple[list[str], list[Any]]:
        scope_clauses, scope_parameters = MemoryObjectRepository.namespace_scope_clauses(
            MemoryObjectRepository.canonical_retrieval_scopes(plan.scope_filter),
            user_persona_id=plan.user_persona_id,
            character_id=plan.character_id if plan.character_id is not None else plan.workspace_id,
            conversation_id=plan.conversation_id,
            remember_across_chats=plan.remember_across_chats,
            incognito=plan.incognito,
            table_alias=alias,
            allow_cross_conversation_chat=CandidateSearch._privacy_sql_filters_disabled(plan),
        )
        if not scope_clauses:
            return [], []
        platform_clause, platform_parameters = MemoryObjectRepository.platform_lock_clause(
            platform_id=str(plan.platform_id or "default"),
            remember_across_devices=plan.remember_across_devices,
            table_alias=alias,
        )
        clauses = [
            "(" + " OR ".join(scope_clauses) + ")",
        ]
        parameters = [*scope_parameters]
        if include_space:
            space_clause, space_parameters = space_visibility_sql_clause(plan, alias=alias)
            clauses.append(space_clause)
            parameters.extend(space_parameters)
        if include_mind:
            mind_clause, mind_parameters = mind_visibility_sql_clause(plan, alias=alias)
            clauses.append(mind_clause)
            parameters.extend(mind_parameters)
        if include_embodiment:
            embodiment_clause, embodiment_parameters = embodiment_visibility_sql_clause(
                plan,
                alias=alias,
            )
            clauses.append(embodiment_clause)
            parameters.extend(embodiment_parameters)
        if include_realm:
            realm_clause, realm_parameters = realm_visibility_sql_clause(
                plan,
                alias=alias,
            )
            clauses.append(realm_clause)
            parameters.extend(realm_parameters)
        clauses.extend(
            [
                CandidateSearch._plan_sensitivity_filter_clause(plan, alias),
                platform_clause,
            ]
        )
        parameters.extend(platform_parameters)
        return clauses, parameters

    @staticmethod
    def _consequence_source_visibility_clauses(
        plan: RetrievalPlan,
        *,
        alias: str,
    ) -> tuple[list[str], list[Any]]:
        scope_clauses, scope_parameters = MemoryObjectRepository.namespace_scope_clauses(
            MemoryObjectRepository.canonical_retrieval_scopes(plan.scope_filter),
            user_persona_id=plan.user_persona_id,
            character_id=plan.character_id if plan.character_id is not None else plan.workspace_id,
            conversation_id=plan.conversation_id,
            remember_across_chats=plan.remember_across_chats,
            incognito=plan.incognito,
            table_alias=alias,
            allow_cross_conversation_chat=CandidateSearch._privacy_sql_filters_disabled(plan),
        )
        source_scope_clause = (
            f"({alias}.scope_canonical = 'chat' "
            f"AND {alias}.user_persona_id IS cc.user_persona_id "
            f"AND {alias}.conversation_id = cc.conversation_id)"
        )
        platform_clause, platform_parameters = MemoryObjectRepository.platform_lock_clause(
            platform_id=str(plan.platform_id or "default"),
            remember_across_devices=plan.remember_across_devices,
            table_alias=alias,
        )
        space_clause, space_parameters = space_visibility_sql_clause(plan, alias=alias)
        mind_clause, mind_parameters = mind_visibility_sql_clause(plan, alias=alias)
        embodiment_clause, embodiment_parameters = embodiment_visibility_sql_clause(plan, alias=alias)
        realm_clause, realm_parameters = realm_visibility_sql_clause(plan, alias=alias)
        return (
            [
                "(" + " OR ".join([*scope_clauses, source_scope_clause]) + ")",
                space_clause,
                mind_clause,
                embodiment_clause,
                realm_clause,
                CandidateSearch._plan_sensitivity_filter_clause(plan, alias),
                platform_clause,
            ],
            [
                *scope_parameters,
                *space_parameters,
                *mind_parameters,
                *embodiment_parameters,
                *realm_parameters,
                *platform_parameters,
            ],
        )

    @staticmethod
    def _plan_sensitivity_filter_clause(plan: RetrievalPlan, alias: str) -> str:
        _assert_safe_alias(alias)
        if CandidateSearch._privacy_sql_filters_disabled(plan):
            return "1 = 1"
        prefix = f"{alias}." if alias else ""
        if plan.allow_private_sensitivity:
            return f"({prefix}sensitivity IN ('public', 'private'))"
        return MemoryObjectRepository.sensitivity_filter_clause(
            gates_enabled=False,
            table_alias=alias,
        )

    @staticmethod
    def _privacy_sql_filters_disabled(plan: RetrievalPlan) -> bool:
        return plan.privacy_enforcement == "off"

    @staticmethod
    def _privacy_ceiling_filter_clause(
        plan: RetrievalPlan,
        *,
        alias: str,
    ) -> tuple[str, list[Any]]:
        _assert_safe_alias(alias)
        if CandidateSearch._privacy_sql_filters_disabled(plan):
            return "1 = 1", []
        prefix = f"{alias}." if alias else ""
        return f"{prefix}privacy_level <= ?", [plan.privacy_ceiling]

    @staticmethod
    def _effective_privacy_ceiling(plan: RetrievalPlan) -> int:
        if CandidateSearch._privacy_sql_filters_disabled(plan):
            return 3
        return plan.privacy_ceiling

    @staticmethod
    def _effective_allow_intimacy_context(plan: RetrievalPlan) -> bool:
        if CandidateSearch._privacy_sql_filters_disabled(plan):
            return True
        return plan.allow_intimacy_context

    @staticmethod
    def _effective_allow_private_sensitivity(plan: RetrievalPlan) -> bool:
        if CandidateSearch._privacy_sql_filters_disabled(plan):
            return False
        return plan.allow_private_sensitivity

    @staticmethod
    def _memory_intimacy_filter_clause(plan: RetrievalPlan, *, alias: str) -> str:
        _assert_safe_alias(alias)
        if CandidateSearch._privacy_sql_filters_disabled(plan):
            return "1 = 1"
        return memory_object_intimacy_sql_clause(
            alias,
            allow_intimacy_context=plan.allow_intimacy_context,
        )

    @staticmethod
    def _coalesced_intimacy_filter_clause(
        plan: RetrievalPlan,
        *,
        primary_alias: str,
        fallback_alias: str,
    ) -> str:
        _assert_safe_alias(primary_alias)
        _assert_safe_alias(fallback_alias)
        if CandidateSearch._privacy_sql_filters_disabled(plan):
            return "1 = 1"
        return coalesced_intimacy_sql_clause(
            primary_alias,
            fallback_alias,
            allow_intimacy_context=plan.allow_intimacy_context,
        )

    @staticmethod
    def _scope_order_case(scope_filter: list[MemoryScope], alias: str = "mo") -> str:
        canonical_filter = MemoryObjectRepository.canonical_retrieval_scopes(scope_filter)
        clauses = [
            f"WHEN {alias}.scope_canonical = '{scope.value}' THEN {index}"
            for index, scope in enumerate(canonical_filter)
        ]
        return "CASE " + " ".join(clauses) + f" ELSE {len(canonical_filter)} END"

    def _temporal_order_parameters(self, plan: RetrievalPlan) -> tuple[str, ...]:
        if plan.temporal_query_range is None:
            return ()
        query_start_minus_horizon = self._subtract_non_negative_hours_clamped(
            plan.temporal_query_range.start,
            self._settings.ephemeral_scoring_hours,
        ).isoformat()
        return (
            plan.temporal_query_range.end.isoformat(),
            plan.temporal_query_range.start.isoformat(),
            query_start_minus_horizon,
            plan.temporal_query_range.end.isoformat(),
            plan.temporal_query_range.start.isoformat(),
        )

    @staticmethod
    def _subtract_non_negative_hours_clamped(value: datetime, hours: int) -> datetime:
        if hours < 0:
            raise ValueError("hours must be non-negative")
        try:
            return value - timedelta(hours=hours)
        except OverflowError:
            return datetime.min.replace(tzinfo=value.tzinfo)

    def _temporal_order_case(self, plan: RetrievalPlan, *, alias: str) -> str:
        if plan.temporal_query_range is None:
            return ""
        return (
            "CASE "
            f"WHEN {alias}.temporal_type = 'unknown' THEN {_TEMPORAL_UNKNOWN_PRIORITY} "
            f"WHEN {alias}.temporal_type = 'ephemeral' "
            f"AND {alias}.valid_from IS NOT NULL "
            f"AND {alias}.valid_from <= ? "
            f"AND ("
            f"({alias}.valid_to IS NOT NULL AND {alias}.valid_to >= ?) "
            f"OR ({alias}.valid_to IS NULL AND {alias}.valid_from >= ?)"
            f") THEN {_TEMPORAL_OVERLAP_PRIORITY} "
            f"WHEN {alias}.temporal_type = 'ephemeral' THEN {_TEMPORAL_NON_OVERLAP_PRIORITY} "
            f"WHEN (({alias}.valid_from IS NULL OR {alias}.valid_from <= ?) "
            f"AND ({alias}.valid_to IS NULL OR {alias}.valid_to >= ?)) THEN {_TEMPORAL_OVERLAP_PRIORITY} "
            f"ELSE {_TEMPORAL_NON_OVERLAP_PRIORITY} END, "
        )

    @staticmethod
    def _coalesced_scope_clauses(
        plan: RetrievalPlan,
        *,
        primary_alias: str,
        fallback_alias: str,
    ) -> tuple[list[str], list[Any]]:
        scope_expr = (
            f"COALESCE({primary_alias}.scope_canonical, {fallback_alias}.scope_canonical, "
            f"{primary_alias}.scope, {fallback_alias}.scope)"
        )
        user_persona_expr = f"COALESCE({primary_alias}.user_persona_id, {fallback_alias}.user_persona_id)"
        character_expr = f"COALESCE({primary_alias}.character_id, {fallback_alias}.character_id)"
        conversation_expr = f"COALESCE({primary_alias}.conversation_id, {fallback_alias}.conversation_id)"

        clauses: list[str] = []
        parameters: list[Any] = []
        for scope in MemoryObjectRepository.canonical_retrieval_scopes(plan.scope_filter):
            if scope is MemoryScope.CHAT:
                clauses.append(
                    f"({scope_expr} = 'chat' AND {user_persona_expr} IS ? AND {conversation_expr} = ?)"
                )
                parameters.extend([plan.user_persona_id, plan.conversation_id])
            elif scope is MemoryScope.CHARACTER:
                expected_character_id = plan.character_id if plan.character_id is not None else plan.workspace_id
                if plan.incognito or not plan.remember_across_chats or expected_character_id is None:
                    continue
                clauses.append(
                    f"({scope_expr} = 'character' AND {user_persona_expr} IS ? AND {character_expr} = ?)"
                )
                parameters.extend([plan.user_persona_id, expected_character_id])
            elif scope is MemoryScope.USER:
                if plan.incognito or not plan.remember_across_chats:
                    continue
                clauses.append(f"({scope_expr} = 'user' AND {user_persona_expr} IS ?)")
                parameters.append(plan.user_persona_id)
        return clauses, parameters

    @staticmethod
    def _coalesced_scope_order_case(
        scope_filter: list[MemoryScope],
        *,
        primary_alias: str,
        fallback_alias: str,
    ) -> str:
        scope_expr = (
            f"COALESCE({primary_alias}.scope_canonical, {fallback_alias}.scope_canonical, "
            f"{primary_alias}.scope, {fallback_alias}.scope)"
        )
        canonical_filter = MemoryObjectRepository.canonical_retrieval_scopes(scope_filter)
        clauses = [
            f"WHEN {scope_expr} = '{scope.value}' THEN {index}"
            for index, scope in enumerate(canonical_filter)
        ]
        return "CASE " + " ".join(clauses) + f" ELSE {len(canonical_filter)} END"

    def _coalesced_temporal_order_case(
        self,
        plan: RetrievalPlan,
        *,
        primary_alias: str,
        fallback_alias: str,
    ) -> str:
        if plan.temporal_query_range is None:
            return ""
        temporal_type_expr = f"COALESCE({primary_alias}.temporal_type, {fallback_alias}.temporal_type)"
        valid_from_expr = f"COALESCE({primary_alias}.valid_from, {fallback_alias}.valid_from)"
        valid_to_expr = f"COALESCE({primary_alias}.valid_to, {fallback_alias}.valid_to)"
        return (
            "CASE "
            f"WHEN {temporal_type_expr} = 'unknown' THEN {_TEMPORAL_UNKNOWN_PRIORITY} "
            f"WHEN {temporal_type_expr} = 'ephemeral' "
            f"AND {valid_from_expr} IS NOT NULL "
            f"AND {valid_from_expr} <= ? "
            f"AND ("
            f"({valid_to_expr} IS NOT NULL AND {valid_to_expr} >= ?) "
            f"OR ({valid_to_expr} IS NULL AND {valid_from_expr} >= ?)"
            f") THEN {_TEMPORAL_OVERLAP_PRIORITY} "
            f"WHEN {temporal_type_expr} = 'ephemeral' THEN {_TEMPORAL_NON_OVERLAP_PRIORITY} "
            f"WHEN (({valid_from_expr} IS NULL OR {valid_from_expr} <= ?) "
            f"AND ({valid_to_expr} IS NULL OR {valid_to_expr} >= ?)) THEN {_TEMPORAL_OVERLAP_PRIORITY} "
            f"ELSE {_TEMPORAL_NON_OVERLAP_PRIORITY} END, "
        )

    @staticmethod
    def _needs_consequence_chain_search(plan: RetrievalPlan) -> bool:
        return plan.consequence_search_enabled

    async def _load_memory_object(
        self,
        memory_id: str,
        user_id: str,
        *,
        conversation_id: str | None,
    ) -> dict[str, Any] | None:
        cursor = await self._connection.execute(
            """
            SELECT *
            FROM memory_objects AS mo
            WHERE mo.id = ?
              AND mo.user_id = ?
              AND mo.archived_by_conversation_id IS NULL
              AND {visibility_clause}
            """.format(visibility_clause=conversation_visibility_clause("mo")),
            (memory_id, user_id, conversation_id),
        )
        row = await cursor.fetchone()
        return _decode_json_columns(row)

    # ------------------------------------------------------------------
    # FTS-backed verbatim evidence channel
    # ------------------------------------------------------------------

    async def _search_verbatim_evidence(
        self,
        *,
        plan: RetrievalPlan,
        user_id: str,
        sub_query: Any,
    ) -> list[dict[str, Any]]:
        """Return ranked conversation windows for a single sub-query.

        Steps:
        1. Run a privacy-filtered FTS query on messages.
        2. Materialize a ``verbatim_evidence_window_size`` conversation window
           around each match. Overlap across neighbouring matches is
           collapsed by keeping the best-ranked seed per window slot.
        3. Shape the window as a pseudo-candidate compatible with the
           downstream fusion / filtering pipeline.
        """
        if not self._verbatim_evidence_search_enabled:
            return []
        if self._verbatim_evidence_search_limit <= 0:
            return []

        channel_scope = self._pick_verbatim_evidence_scope(plan)
        if channel_scope is None:
            return []

        # Verbatim evidence is strongest on the precise FTS variant.
        # Planner order already goes from most precise to broader
        # rewrites. Exact recall still prefers that strongest query
        # first, but a multilingual bridge can make the precise rewrite
        # too strict for transcript evidence search. In that case, keep
        # falling back to broader rewrites until the bounded window limit
        # is filled.
        fts_queries = self._verbatim_evidence_fts_queries(plan, sub_query)
        if not fts_queries:
            return []

        aggregated: dict[str, dict[str, Any]] = {}
        queries_to_try = fts_queries
        if plan.exact_recall_mode:
            queries_to_try = []
            for index, fts_query in enumerate(fts_queries):
                queries_to_try.append(fts_query)
                if index == 0:
                    continue
                # Exact recall may need a bounded fallback when the
                # strictest rewrite over-constrains raw transcript
                # search and returns no usable windows.
                if aggregated:
                    break
        active_conversation_only = (
            not self._privacy_sql_filters_disabled(plan)
            and not self._allow_cross_conversation_verbatim_evidence(plan)
        )
        for fts_query in queries_to_try:
            rows = await self._message_repository.search_messages_with_privacy(
                user_id=user_id,
                query=fts_query,
                privacy_ceiling=self._effective_privacy_ceiling(plan),
                limit=self._verbatim_evidence_search_limit,
                allow_conversation_id=plan.conversation_id,
                active_conversation_only=active_conversation_only,
                include_pending_confirmation_sources=self._privacy_sql_filters_disabled(plan),
            )
            for row in rows:
                for window_candidate in await self._verbatim_evidence_search_window_candidates(
                    seed_row=row,
                    plan=plan,
                    user_id=user_id,
                    channel_scope=channel_scope,
                ):
                    if not self._matches_plan_filters(window_candidate, plan):
                        continue
                    window_id = str(window_candidate["id"])
                    existing = aggregated.get(window_id)
                    if existing is None or float(window_candidate.get("fts_rank", 0.0)) < float(
                        existing.get("fts_rank", 0.0)
                    ):
                        aggregated[window_id] = window_candidate
            if (
                plan.exact_recall_mode
                and len(aggregated) >= self._verbatim_evidence_search_limit
            ):
                break

        ordered = sorted(
            aggregated.values(),
            key=lambda candidate: (
                float(candidate.get("fts_rank", 0.0)),
                str(candidate["id"]),
            ),
        )
        return self._assign_position_ranks(ordered[: self._verbatim_evidence_search_limit])

    @staticmethod
    def _verbatim_evidence_fts_queries(plan: RetrievalPlan, sub_query: Any) -> list[str]:
        queries: list[str] = []
        seen: set[str] = set()
        for fts_query in list(sub_query.fts_queries or []):
            if fts_query in seen:
                continue
            seen.add(fts_query)
            queries.append(fts_query)
        if plan.exact_recall_mode and plan.original_query:
            try:
                original_query_rewrites = build_retrieval_fts_queries(plan.original_query)
            except ValueError:
                original_query_rewrites = []
            for fts_query in original_query_rewrites:
                if fts_query in seen:
                    continue
                seen.add(fts_query)
                queries.append(fts_query)
        return queries

    async def _verbatim_evidence_search_window_candidates(
        self,
        *,
        seed_row: dict[str, Any],
        plan: RetrievalPlan,
        user_id: str,
        channel_scope: MemoryScope,
    ) -> list[dict[str, Any]]:
        """Build centered and exact slot-fill follow-up evidence windows."""
        seed_seq = int(seed_row["seq"])
        window_specs = [(seed_seq, self._verbatim_evidence_window_size, "centered", 0.0)]
        if plan.exact_recall_mode and plan.query_type == "slot_fill":
            follow_up_window_size = min(5, self._verbatim_evidence_window_size + 2)
            window_specs.append(
                (seed_seq + 1, follow_up_window_size, "follow_up", -0.000001)
            )

        candidates: list[dict[str, Any]] = []
        seen_window_ids: set[str] = set()
        for center_seq, window_size, window_variant, rank_adjustment in window_specs:
            window_candidate = await self._build_verbatim_evidence_search_window_candidate(
                seed_row=seed_row,
                plan=plan,
                user_id=user_id,
                channel_scope=channel_scope,
                center_seq=center_seq,
                window_size=window_size,
                window_variant=window_variant,
                rank_adjustment=rank_adjustment,
            )
            if window_candidate is None:
                continue
            window_id = str(window_candidate["id"])
            if window_id in seen_window_ids:
                continue
            seen_window_ids.add(window_id)
            candidates.append(window_candidate)
        return candidates

    async def _build_verbatim_evidence_search_window_candidate(
        self,
        *,
        seed_row: dict[str, Any],
        plan: RetrievalPlan,
        user_id: str,
        channel_scope: MemoryScope,
        center_seq: int | None = None,
        window_size: int | None = None,
        window_variant: str = "centered",
        rank_adjustment: float = 0.0,
    ) -> dict[str, Any] | None:
        """Shape a seed message row into a conversation-window candidate."""
        conversation_id = str(seed_row["conversation_id"])
        seed_seq = int(seed_row["seq"])
        window_messages = await self._message_repository.fetch_message_window(
            conversation_id=conversation_id,
            user_id=user_id,
            center_seq=seed_seq if center_seq is None else center_seq,
            window_size=(
                self._verbatim_evidence_window_size
                if window_size is None
                else window_size
            ),
        )
        if not window_messages:
            return None
        window_messages = [
            self._message_with_seed_snippet(message, seed_row=seed_row, seed_seq=seed_seq)
            for message in window_messages
        ]
        window_message_ids = [str(message["id"]) for message in window_messages]
        metadata_rows = await self._source_message_metadata_rows(
            user_id=user_id,
            message_ids=window_message_ids,
        )
        if not self._privacy_sql_filters_disabled(plan):
            window_messages = await self._gate_pending_confirmation_window_messages(
                messages=window_messages,
                pending_ids=metadata_rows.pending_message_ids(),
            )
        conversation_mode_id = str(seed_row.get("conversation_assistant_mode_id") or "")
        # Scope mapping guards cross-conversation leakage for the
        # downstream scope filter. Mode is retrieval-profile tuning, not
        # a namespace boundary.
        resolved_scope = self._resolve_verbatim_evidence_window_scope(
            channel_scope=channel_scope,
            plan=plan,
            conversation_id=conversation_id,
        )
        if resolved_scope is None:
            return None

        start_seq = int(window_messages[0]["seq"])
        end_seq = int(window_messages[-1]["seq"])
        window_id = (
            f"{_VERBATIM_EVIDENCE_WINDOW_ID_PREFIX}"
            f"{conversation_id}_{start_seq}_{end_seq}"
        )
        canonical_text = self._format_verbatim_evidence_window_text(
            window_messages,
            include_skipped_raw=(
                plan.exact_recall_mode
                or plan.raw_context_access_mode in {"skipped_raw", "verbatim"}
            ),
        )
        occurred_at = (
            window_messages[-1].get("occurred_at")
            or window_messages[-1].get("created_at")
        )
        source_message_ids = [
            str(message["id"])
            for message in window_messages
            if not self._message_is_pending_confirmation(message)
        ]
        (
            privacy_level,
            high_risk_metadata,
            intimacy_boundary,
            intimacy_boundary_confidence,
        ) = metadata_rows.metadata_for_source_message_ids(
            source_message_ids=source_message_ids,
        )
        source_window_start = str(
            window_messages[0].get("occurred_at")
            or window_messages[0].get("created_at")
            or ""
        )
        source_window_end = str(
            window_messages[-1].get("occurred_at")
            or window_messages[-1].get("created_at")
            or ""
        )
        payload: dict[str, Any] = {
            "source_message_ids": source_message_ids,
            "source_kind_variant": "conversation_window",
            "window_start_seq": start_seq,
            "window_end_seq": end_seq,
            "window_size": len(window_messages),
            "verbatim_evidence_window_variant": window_variant,
            "source_message_window_start_occurred_at": source_window_start,
            "source_message_window_end_occurred_at": source_window_end,
        }
        space_id, space_boundary_mode = self._source_space_fields(
            plan,
            [seed_row, *window_messages],
            space_id_key="space_id",
            boundary_mode_key="message_space_boundary_mode",
        )
        if space_id is not None:
            payload["space_boundary"] = {
                "active_space_id": space_id,
                "boundary_mode": space_boundary_mode,
            }
        memory_owner_id, source_mind_id = self._source_mind_fields(
            plan,
            [seed_row, *window_messages],
        )
        if memory_owner_id is not None:
            payload["mind_perspective"] = {
                "memory_owner_id": memory_owner_id,
                "source_mind_id": source_mind_id,
                "mind_topology": (
                    plan.mind_topology.value
                    if hasattr(plan.mind_topology, "value")
                    else str(plan.mind_topology)
                ),
            }
        embodiment_id = self._source_embodiment_id(
            plan,
            [seed_row, *window_messages],
            keys=("active_embodiment_id", "embodiment_id"),
        )
        if embodiment_id is not None:
            payload["embodiment"] = {
                "active_embodiment_id": embodiment_id,
            }
        realm_id = self._source_realm_id(
            plan,
            [seed_row, *window_messages],
            keys=("active_realm_id", "realm_id"),
        )
        if realm_id is not None:
            payload["realm"] = {
                "active_realm_id": realm_id,
            }
        fts_rank = float(seed_row.get("rank") or 0.0) + rank_adjustment
        # Phase 7 only permits raw/verbatim evidence from the active chat.
        # The window is user-visible transcript evidence rather than a
        # durable memory fact, so namespace gating happens through the active
        # conversation check while high-risk payload metadata still controls
        # redaction.
        sensitivity = "public"
        return {
            "_rowid": None,
            "id": window_id,
            "user_id": user_id,
            "workspace_id": plan.workspace_id,
            "conversation_id": conversation_id,
            "assistant_mode_id": conversation_mode_id,
            "user_persona_id": plan.user_persona_id,
            "platform_id": plan.platform_id,
            "character_id": plan.character_id if plan.character_id is not None else plan.workspace_id,
            "space_id": space_id,
            "space_boundary_mode": space_boundary_mode,
            "memory_owner_id": memory_owner_id,
            "source_mind_id": source_mind_id,
            "embodiment_id": embodiment_id,
            "realm_id": realm_id,
            "object_type": MemoryObjectType.EVIDENCE.value,
            "scope": resolved_scope.value,
            "scope_canonical": _canonical_scope_value(resolved_scope.value),
            "canonical_text": canonical_text,
            "payload_json": payload,
            "source_kind": MemorySourceKind.VERBATIM.value,
            "confidence": 1.0,
            "stability": 0.5,
            "vitality": 0.0,
            "maya_score": 0.0,
            "privacy_level": privacy_level,
            **high_risk_metadata,
            "intimacy_boundary": intimacy_boundary,
            "intimacy_boundary_confidence": intimacy_boundary_confidence,
            "temporal_type": "unknown",
            "valid_from": None,
            "valid_to": None,
            "status": MemoryStatus.ACTIVE.value,
            "sensitivity": sensitivity,
            "themes_json": [],
            "platform_locked": 0,
            "platform_id_lock": None,
            "created_at": str(seed_row.get("created_at")),
            "updated_at": str(occurred_at or seed_row.get("created_at")),
            "rank": fts_rank,
            "fts_rank": fts_rank,
            "retrieval_level": 0,
            "is_verbatim_evidence_window": True,
            "verbatim_evidence_window_conversation_id": conversation_id,
            "verbatim_evidence_window_start_seq": start_seq,
            "verbatim_evidence_window_end_seq": end_seq,
            "verbatim_evidence_window_variant": window_variant,
            "verbatim_evidence_window_message_ids": source_message_ids,
        }

    async def _gate_pending_confirmation_window_messages(
        self,
        *,
        messages: list[dict[str, Any]],
        pending_ids: set[str],
    ) -> list[dict[str, Any]]:
        if not pending_ids:
            return messages
        gated_messages: list[dict[str, Any]] = []
        for message in messages:
            message_id = str(message.get("id") or "")
            if message_id not in pending_ids:
                gated_messages.append(message)
                continue
            gated = dict(message)
            gated.update(
                {
                    "include_raw": False,
                    "skip_by_default": True,
                    "pending_user_confirmation": True,
                    "policy_reason": "pending_user_confirmation",
                }
            )
            gated_messages.append(gated)
        return gated_messages

    async def _source_message_metadata_rows(
        self,
        *,
        user_id: str,
        message_ids: list[str],
    ) -> _SourceMessageMetadataRows:
        normalized_message_ids = [
            str(message_id).strip()
            for message_id in message_ids
            if str(message_id).strip()
        ]
        if not normalized_message_ids:
            return _SourceMessageMetadataRows([])
        placeholders = ", ".join("?" for _ in normalized_message_ids)
        cursor = await self._connection.execute(
            """
            SELECT
                CAST(source_ids.value AS TEXT) AS message_id,
                mo.status,
                mo.privacy_level,
                mo.memory_category,
                mo.preserve_verbatim,
                mo.intimacy_boundary,
                mo.intimacy_boundary_confidence
            FROM memory_objects AS mo
            JOIN json_each(
                json_extract(mo.payload_json, '$.source_message_ids')
            ) AS source_ids ON 1 = 1
            WHERE mo.user_id = ?
              AND CAST(source_ids.value AS TEXT) IN ({placeholders})
            """.format(placeholders=placeholders),
            (user_id, *normalized_message_ids),
        )
        return _SourceMessageMetadataRows([dict(row) for row in await cursor.fetchall()])

    @staticmethod
    def _build_artifact_chunk_candidate(
        row: dict[str, Any],
        *,
        plan: RetrievalPlan,
        sub_query_text: str,
    ) -> dict[str, Any]:
        artifact_metadata = row.get("artifact_metadata_json") or {}
        if not isinstance(artifact_metadata, dict):
            artifact_metadata = {}
        content_kind = str(row.get("kind") or "parsed")
        canonical_text = str(row.get("text") or "").strip()
        artifact_id = str(row["artifact_id"])
        chunk_id = str(row["id"])
        source_kind = str(row.get("source_kind") or "extracted")
        artifact_type = str(row.get("artifact_type") or "other")
        preserve_verbatim = bool(row.get("preserve_verbatim"))
        source_window_start = artifact_metadata.get("source_window_start")
        source_window_end = artifact_metadata.get("source_window_end")
        boundary_rows = [
            {
                "intimacy_boundary": row.get("artifact_intimacy_boundary"),
                "intimacy_boundary_confidence": row.get("artifact_intimacy_boundary_confidence"),
            },
            {
                "intimacy_boundary": row.get("intimacy_boundary"),
                "intimacy_boundary_confidence": row.get("intimacy_boundary_confidence"),
            },
        ]
        boundary = strongest_intimacy_boundary(boundary_rows)
        intimacy_boundary = boundary.value
        intimacy_boundary_confidence = max(
            (
                float(item.get("intimacy_boundary_confidence", 0.0) or 0.0)
                for item in boundary_rows
                if str(item.get("intimacy_boundary") or "ordinary") == boundary.value
            ),
            default=0.0,
        )
        privacy_level = int(row.get("privacy_level", 0))
        derived_sensitivity = "public" if privacy_level <= 1 and intimacy_boundary == "ordinary" else "private"
        sensitivity = CandidateSearch._strictest_sensitivity(
            str(row.get("artifact_sensitivity") or "unknown"),
            derived_sensitivity,
        )
        space_id, space_boundary_mode = CandidateSearch._source_space_fields(
            plan,
            [row],
            space_id_key="source_message_space_id",
            boundary_mode_key="source_message_space_boundary_mode",
        )
        space_payload = (
            {"active_space_id": space_id, "boundary_mode": space_boundary_mode}
            if space_id is not None
            else None
        )
        memory_owner_id, source_mind_id = CandidateSearch._source_mind_fields(
            plan,
            [row],
            owner_id_keys=("artifact_memory_owner_id", "source_message_active_mind_id", "active_mind_id"),
            source_id_keys=("artifact_source_mind_id", "source_message_source_mind_id", "source_mind_id"),
        )
        mind_payload = (
            {
                "memory_owner_id": memory_owner_id,
                "source_mind_id": source_mind_id,
                "mind_topology": (
                    plan.mind_topology.value
                    if hasattr(plan.mind_topology, "value")
                    else str(plan.mind_topology)
                ),
            }
            if memory_owner_id is not None
            else None
        )
        embodiment_id = CandidateSearch._source_embodiment_id(
            plan,
            [row],
            keys=("artifact_embodiment_id", "source_message_active_embodiment_id", "active_embodiment_id"),
        )
        embodiment_payload = (
            {"active_embodiment_id": embodiment_id}
            if embodiment_id is not None
            else None
        )
        realm_id = CandidateSearch._source_realm_id(
            plan,
            [row],
            keys=("artifact_realm_id", "source_message_active_realm_id", "active_realm_id"),
        )
        realm_payload = (
            {"active_realm_id": realm_id}
            if realm_id is not None
            else None
        )
        candidate: dict[str, Any] = {
            "_rowid": None,
            "id": chunk_id,
            "artifact_id": artifact_id,
            "artifact_chunk_id": chunk_id,
            "artifact_type": artifact_type,
            "artifact_source_kind": source_kind,
            "artifact_status": str(row.get("status") or "ready"),
            "artifact_title": row.get("title"),
            "artifact_filename": row.get("filename"),
            "artifact_source_ref": row.get("source_ref"),
            "artifact_summary_text": row.get("artifact_summary_text"),
            "artifact_index_text": row.get("artifact_index_text"),
            "artifact_page_count": row.get("page_count"),
            "artifact_preserve_verbatim": preserve_verbatim,
            "user_id": str(row["user_id"]),
            "workspace_id": row.get("workspace_id"),
            "conversation_id": row.get("conversation_id"),
            "message_id": row.get("message_id"),
            "user_persona_id": plan.user_persona_id,
            "platform_id": plan.platform_id,
            "character_id": plan.character_id if plan.character_id is not None else plan.workspace_id,
            "space_id": space_id,
            "space_boundary_mode": space_boundary_mode,
            "memory_owner_id": memory_owner_id,
            "source_mind_id": source_mind_id,
            "embodiment_id": embodiment_id,
            "realm_id": realm_id,
            "object_type": MemoryObjectType.EVIDENCE.value,
            "scope": CandidateSearch._scope_for_artifact_row(row, plan),
            "scope_canonical": MemoryScope.CHAT.value,
            "canonical_text": canonical_text,
            "intimacy_boundary": intimacy_boundary,
            "intimacy_boundary_confidence": intimacy_boundary_confidence,
            "payload_json": {
                "artifact_id": artifact_id,
                "artifact_chunk_id": chunk_id,
                "artifact_type": artifact_type,
                "artifact_source_kind": source_kind,
                "artifact_status": str(row.get("status") or "ready"),
                "artifact_title": row.get("title"),
                "artifact_filename": row.get("filename"),
                "artifact_source_ref": row.get("source_ref"),
                "artifact_summary_text": row.get("artifact_summary_text"),
                "artifact_index_text": row.get("artifact_index_text"),
                "artifact_page_count": row.get("page_count"),
                "content_kind": content_kind,
                "source_message_ids": [str(row["message_id"])] if row.get("message_id") else [],
                "sub_query_text": sub_query_text,
                "source_window_start_occurred_at": source_window_start,
                "source_window_end_occurred_at": source_window_end,
                "intimacy_boundary": intimacy_boundary,
                "intimacy_boundary_confidence": intimacy_boundary_confidence,
                **({"space_boundary": space_payload} if space_payload is not None else {}),
                **({"mind_perspective": mind_payload} if mind_payload is not None else {}),
                **({"embodiment": embodiment_payload} if embodiment_payload is not None else {}),
                **({"realm": realm_payload} if realm_payload is not None else {}),
            },
            "source_kind": MemorySourceKind.VERBATIM.value
            if preserve_verbatim
            else MemorySourceKind.EXTRACTED.value,
            "confidence": 1.0 if preserve_verbatim else 0.9,
            "stability": 0.5,
            "vitality": 0.0,
            "maya_score": 0.0,
            "privacy_level": privacy_level,
            "temporal_type": "unknown",
            "valid_from": None,
            "valid_to": None,
            "status": "active",
            "sensitivity": sensitivity,
            "themes_json": [],
            "platform_locked": 0,
            "platform_id_lock": None,
            "created_at": str(row.get("artifact_created_at") or row.get("created_at")),
            "updated_at": str(row.get("artifact_updated_at") or row.get("updated_at")),
            "rank": float(row.get("rank") or 0.0),
            "fts_rank": float(row.get("rank") or 0.0),
            "retrieval_level": 0,
            "is_artifact_chunk": True,
            "artifact_chunk_chunk_index": int(row.get("chunk_index", 0)),
            "artifact_chunk_kind": content_kind,
        }
        candidate["retrieval_sources"] = ["artifact_chunk"]
        candidate["channel_ranks"] = {
            "verbatim_pin": None,
            "artifact_chunk": 1,
            "fts": None,
            "fact_facet": None,
            "embedding": None,
            "consequence": None,
            "verbatim_evidence_search": None,
        }
        candidate["retrieval_source"] = "artifact_chunk"
        candidate["matched_sub_queries"] = [sub_query_text]
        return candidate

    @staticmethod
    def _source_space_fields(
        plan: RetrievalPlan,
        rows: list[dict[str, Any]],
        *,
        space_id_key: str,
        boundary_mode_key: str,
    ) -> tuple[str | None, str | None]:
        space_id = next(
            (
                str(row.get(space_id_key)).strip()
                for row in rows
                if row.get(space_id_key) is not None and str(row.get(space_id_key)).strip()
            ),
            None,
        )
        if space_id is None:
            space_id = plan.active_space_id
        if space_id is None:
            return None, None
        boundary_mode = next(
            (
                str(row.get(boundary_mode_key)).strip()
                for row in rows
                if row.get(boundary_mode_key) is not None
                and str(row.get(boundary_mode_key)).strip()
            ),
            None,
        )
        if boundary_mode is None and space_id == plan.active_space_id:
            boundary_mode = (
                plan.active_space_boundary_mode.value
                if isinstance(plan.active_space_boundary_mode, SpaceBoundaryMode)
                else plan.active_space_boundary_mode
            )
        try:
            resolved_mode = SpaceBoundaryMode(str(boundary_mode or SpaceBoundaryMode.FOCUS.value)).value
        except ValueError:
            resolved_mode = SpaceBoundaryMode.FOCUS.value
        return space_id, resolved_mode

    @staticmethod
    def _source_mind_fields(
        plan: RetrievalPlan,
        rows: list[dict[str, Any]],
        *,
        owner_id_keys: tuple[str, ...] = ("active_mind_id", "memory_owner_id"),
        source_id_keys: tuple[str, ...] = ("source_mind_id", "active_mind_id"),
    ) -> tuple[str | None, str | None]:
        owner_id = CandidateSearch._first_text_from_rows(rows, owner_id_keys)
        source_id = CandidateSearch._first_text_from_rows(rows, source_id_keys)
        if owner_id is None:
            owner_id = plan.active_mind_id
        if source_id is None:
            source_id = owner_id or plan.active_mind_id
        return owner_id, source_id

    @staticmethod
    def _source_embodiment_id(
        plan: RetrievalPlan,
        rows: list[dict[str, Any]],
        *,
        keys: tuple[str, ...],
    ) -> str | None:
        embodiment_id = CandidateSearch._first_text_from_rows(rows, keys)
        if embodiment_id is None:
            embodiment_id = plan.active_embodiment_id
        return embodiment_id

    @staticmethod
    def _source_realm_id(
        plan: RetrievalPlan,
        rows: list[dict[str, Any]],
        *,
        keys: tuple[str, ...],
    ) -> str | None:
        realm_id = CandidateSearch._first_text_from_rows(rows, keys)
        if realm_id is None:
            realm_id = plan.active_realm_id
        return realm_id

    @staticmethod
    def _first_text_from_rows(
        rows: list[dict[str, Any]],
        keys: tuple[str, ...],
    ) -> str | None:
        for row in rows:
            for key in keys:
                value = row.get(key)
                if value is None:
                    continue
                normalized = str(value).strip()
                if normalized:
                    return normalized
        return None

    @staticmethod
    def _strictest_sensitivity(*values: str) -> str:
        resolved = "unknown"
        for value in values:
            candidate = str(value or "unknown")
            if _SENSITIVITY_RANK.get(candidate, -1) > _SENSITIVITY_RANK[resolved]:
                resolved = candidate
        return resolved

    @staticmethod
    def _message_with_seed_snippet(
        message: dict[str, Any],
        *,
        seed_row: dict[str, Any],
        seed_seq: int,
    ) -> dict[str, Any]:
        if int(message.get("seq", -1)) != seed_seq:
            return message
        snippet = str(seed_row.get("fts_snippet") or "").strip()
        if not snippet:
            return message
        enriched = dict(message)
        enriched["fts_snippet"] = snippet
        return enriched

    @classmethod
    def _format_verbatim_evidence_window_text(
        cls,
        messages: list[dict[str, Any]],
        *,
        include_skipped_raw: bool = False,
    ) -> str:
        """Format a conversation window as a compact transcript block."""
        lines: list[str] = []
        for message in messages:
            role = str(message.get("role") or "user")
            if cls._message_is_pending_confirmation(message):
                text = cls._message_placeholder_text(message)
            elif cls._message_should_use_placeholder(message):
                if include_skipped_raw and cls._message_can_include_skipped_raw(message):
                    text = cls._message_skipped_raw_text(message)
                else:
                    text = cls._message_placeholder_text(message)
            else:
                text = str(message.get("text") or "").strip()
            if not text:
                continue
            lines.append(f"[{role}] {text}")
        return "\n".join(lines)

    @classmethod
    def _message_should_use_placeholder(cls, message: dict[str, Any]) -> bool:
        return cls._truthy(message.get("skip_by_default")) and not cls._truthy(
            message.get("include_raw"),
            default=True,
        )

    @classmethod
    def _message_can_include_skipped_raw(cls, message: dict[str, Any]) -> bool:
        if cls._message_is_pending_confirmation(message):
            return False
        return str(message.get("policy_reason") or "") in {
            "artifact_backed",
            "mechanical_size_threshold",
        }

    @classmethod
    def _message_skipped_raw_text(cls, message: dict[str, Any]) -> str:
        text = str(message.get("text") or "").strip()
        policy_reason = str(message.get("policy_reason") or "skip_by_default")
        if len(text) <= _SKIPPED_RAW_VERBATIM_MAX_CHARS:
            return text
        snippet = " ".join(str(message.get("fts_snippet") or "").split()).strip()
        if snippet:
            excerpt = snippet[:_SKIPPED_RAW_VERBATIM_SNIPPET_MAX_CHARS].strip()
            return (
                "[Skipped raw message excerpt | "
                f"seq={message.get('seq', '?')} policy={policy_reason}]\n"
                f"{excerpt}"
            )
        return (
            text[:_SKIPPED_RAW_VERBATIM_MAX_CHARS].rstrip()
            + "\n[Truncated skipped raw message | "
            f"seq={message.get('seq', '?')} policy={policy_reason}]"
        )

    @classmethod
    def _message_is_pending_confirmation(cls, message: dict[str, Any]) -> bool:
        return cls._truthy(message.get("pending_user_confirmation"))

    @classmethod
    def _message_placeholder_text(cls, message: dict[str, Any]) -> str:
        if cls._message_is_pending_confirmation(message):
            seq = str(message.get("seq") if message.get("seq") is not None else "?")
            role = str(message.get("role") or "user")
            content_kind = str(message.get("content_kind") or "text")
            return (
                f"[Skipped message | seq={seq} role={role} "
                f"kind={content_kind} policy=pending_user_confirmation]"
            )
        existing = " ".join(str(message.get("context_placeholder") or "").split())[
            :300
        ].strip()
        if existing:
            return existing
        message_id = str(message.get("id") or f"msg_{message.get('seq', '?')}")
        seq = str(message.get("seq") if message.get("seq") is not None else "?")
        role = str(message.get("role") or "user")
        content_kind = str(message.get("content_kind") or "text")
        policy_reason = str(message.get("policy_reason") or "skip_by_default")
        return (
            f"[Skipped message | id={message_id} seq={seq} role={role} "
            f"kind={content_kind} policy={policy_reason} ref={message_id}]"
        )

    @staticmethod
    def _truthy(value: Any, *, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if not normalized:
                return default
            return normalized in {"1", "true", "yes", "on"}
        return bool(value)

    @staticmethod
    def _scope_for_artifact_row(row: dict[str, Any], plan: RetrievalPlan) -> str | None:
        workspace_id = row.get("workspace_id")
        conversation_id = str(row.get("conversation_id") or "")
        if (
            MemoryScope.CONVERSATION in plan.scope_filter
            and conversation_id == plan.conversation_id
        ):
            return MemoryScope.CONVERSATION.value
        if (
            MemoryScope.EPHEMERAL_SESSION in plan.scope_filter
            and conversation_id == plan.conversation_id
        ):
            return MemoryScope.EPHEMERAL_SESSION.value
        if (
            MemoryScope.WORKSPACE in plan.scope_filter
            and workspace_id == plan.workspace_id
        ):
            return MemoryScope.WORKSPACE.value
        if (
            MemoryScope.GLOBAL_USER in plan.scope_filter
            and workspace_id is None
            and not conversation_id
        ):
            return MemoryScope.GLOBAL_USER.value
        return None

    @staticmethod
    def _pick_verbatim_evidence_scope(plan: RetrievalPlan) -> MemoryScope | None:
        """Pick the narrowest safe scope for raw transcript windows."""
        allowed = set(plan.scope_filter)
        if (
            CandidateSearch._allow_cross_conversation_verbatim_evidence(plan)
            and (
                MemoryScope.GLOBAL_USER in allowed
                or MemoryScope.ASSISTANT_MODE in allowed
                or MemoryScope.USER in allowed
            )
        ):
            return MemoryScope.GLOBAL_USER
        if MemoryScope.CONVERSATION in allowed:
            return MemoryScope.CONVERSATION
        if MemoryScope.EPHEMERAL_SESSION in allowed:
            return MemoryScope.EPHEMERAL_SESSION
        if MemoryScope.CHAT in allowed:
            return MemoryScope.CHAT
        return None

    @staticmethod
    def _resolve_verbatim_evidence_window_scope(
        *,
        channel_scope: MemoryScope,
        plan: RetrievalPlan,
        conversation_id: str,
    ) -> MemoryScope | None:
        """Map an evidence window to a scope consistent with plan filtering.

        The returned scope must satisfy ``_candidate_matches_scope`` so
        that windows survive the post-fusion filter. Narrower scopes
        require a matching conversation identifier; wider scopes
        (``global_user``) accept any conversation that already cleared
        the SQL-layer privacy check.
        """
        if conversation_id == plan.conversation_id:
            canonical_allowed = set(
                MemoryObjectRepository.canonical_retrieval_scopes(plan.scope_filter)
            )
            if MemoryScope.CHAT in canonical_allowed:
                return MemoryScope.CONVERSATION
        if channel_scope in {MemoryScope.CONVERSATION, MemoryScope.EPHEMERAL_SESSION, MemoryScope.CHAT}:
            if conversation_id != plan.conversation_id:
                if CandidateSearch._privacy_sql_filters_disabled(plan):
                    return MemoryScope.CONVERSATION
                return None
            if channel_scope is MemoryScope.EPHEMERAL_SESSION:
                return MemoryScope.EPHEMERAL_SESSION
            return MemoryScope.CONVERSATION
        if channel_scope in {MemoryScope.GLOBAL_USER, MemoryScope.ASSISTANT_MODE, MemoryScope.USER}:
            if plan.incognito or not plan.remember_across_chats:
                return None
            return MemoryScope.GLOBAL_USER
        return None

    @staticmethod
    def _allow_cross_conversation_verbatim_evidence(plan: RetrievalPlan) -> bool:
        if not plan.exact_recall_mode:
            return False
        if plan.incognito or not plan.remember_across_chats:
            return False
        return (
            plan.query_type == "broad_list"
            or (plan.query_type == "slot_fill" and bool(plan.exact_facets))
            or len(plan.sub_query_plans) > 1
            or len(plan.exact_facets) > 1
        )

    @staticmethod
    def _dedupe_verbatim_evidence_windows_against_memories(
        aggregated: dict[str, dict[str, Any]],
        *,
        plan: RetrievalPlan,
    ) -> None:
        """Drop evidence-window candidates already covered by direct memory.

        An evidence window is considered covered when any of its source
        message ids appears in ``payload_json.source_message_ids`` of a direct
        non-window candidate that also scored for the same sub-query. Summary
        views and fact-facet pointers are deliberately not treated as full
        substitutes: they preserve recall and continuity, but they can omit
        exact names, dates, list members, and surrounding utterance context
        that verbatim windows are meant to rescue.
        """
        if plan.exact_recall_mode or plan.raw_context_access_mode == "verbatim":
            return

        covered_message_ids: set[str] = set()
        for candidate in aggregated.values():
            if candidate.get("is_verbatim_evidence_window"):
                continue
            if candidate.get("is_fact_facet_candidate"):
                continue
            if CandidateSearch._is_summary_like_evidence_coverage(candidate):
                continue
            payload = candidate.get("payload_json")
            if not isinstance(payload, dict):
                continue
            for message_id in payload.get("source_message_ids") or []:
                covered_message_ids.add(str(message_id))

        if not covered_message_ids:
            return

        to_drop: list[str] = []
        for window_id, candidate in aggregated.items():
            if not candidate.get("is_verbatim_evidence_window"):
                continue
            window_message_ids = {
                str(message_id)
                for message_id in candidate.get("verbatim_evidence_window_message_ids") or []
            }
            if window_message_ids & covered_message_ids:
                to_drop.append(window_id)
        for window_id in to_drop:
            aggregated.pop(window_id, None)

    @staticmethod
    def _is_summary_like_evidence_coverage(candidate: dict[str, Any]) -> bool:
        return (
            str(candidate.get("object_type") or "") == MemoryObjectType.SUMMARY_VIEW.value
            or str(candidate.get("source_kind") or "") == MemorySourceKind.SUMMARIZED.value
        )

    async def _annotate_realm_bridge_modes(
        self,
        rows: list[dict[str, Any]],
        plan: RetrievalPlan,
        *,
        realm_keys: tuple[str, ...] = ("realm_id", "active_realm_id"),
    ) -> None:
        await annotate_realm_bridge_modes_for_rows(
            self._connection,
            rows,
            active_realm_id=plan.active_realm_id,
            realm_keys=realm_keys,
        )

    async def _annotate_overseer_grants(
        self,
        rows: list[dict[str, Any]],
        plan: RetrievalPlan,
    ) -> None:
        await annotate_overseer_grants_for_rows(
            self._connection,
            rows,
            active_mind_id=plan.active_mind_id,
            mind_topology=plan.mind_topology,
        )

    @classmethod
    def _matches_plan_filters(cls, candidate: dict[str, Any], plan: RetrievalPlan) -> bool:
        if not cls._privacy_sql_filters_disabled(plan):
            if int(candidate.get("privacy_level", 0)) > plan.privacy_ceiling:
                return False
            sensitivity = str(candidate.get("sensitivity") or "unknown")
            if sensitivity != "public" and not (
                sensitivity == "private" and plan.allow_private_sensitivity
            ):
                return False
        if not cls._candidate_matches_platform(candidate, plan):
            return False
        if not cls._privacy_sql_filters_disabled(
            plan
        ) and not candidate_allows_intimacy_boundary(
            candidate,
            allow_intimacy_context=plan.allow_intimacy_context,
        ):
            return False
        if not candidate_allows_space_boundary(candidate, plan):
            return False
        if not candidate_allows_mind_boundary(candidate, plan):
            return False
        if not candidate_allows_embodiment_boundary(candidate, plan):
            return False
        if not candidate_allows_realm_boundary(candidate, plan):
            return False
        if candidate.get("status") not in {status.value for status in plan.status_filter}:
            return False
        if not cls._candidate_matches_retrieval_levels(candidate, plan):
            return False
        return cls._candidate_matches_scope(candidate, plan)

    @staticmethod
    def _candidate_matches_platform(candidate: dict[str, Any], plan: RetrievalPlan) -> bool:
        platform_id = str(plan.platform_id or "default")
        platform_locked = bool(int(candidate.get("platform_locked") or 0))
        platform_id_lock = candidate.get("platform_id_lock")
        if platform_locked:
            return platform_id_lock == platform_id
        if plan.remember_across_devices:
            return True
        return candidate.get("platform_id") == platform_id

    @staticmethod
    def _candidate_matches_retrieval_levels(candidate: dict[str, Any], plan: RetrievalPlan) -> bool:
        if candidate.get("object_type") != "summary_view":
            return 0 in plan.retrieval_levels
        payload_json = candidate.get("payload_json") or {}
        if not isinstance(payload_json, dict):
            return False
        hierarchy_level = int(payload_json.get("hierarchy_level", -1))
        summary_kind = str(payload_json.get("summary_kind", "")).strip()
        if hierarchy_level == 0:
            return 0 in plan.retrieval_levels and summary_kind == SummaryViewKind.CONVERSATION_CHUNK.value
        if hierarchy_level == 1:
            return 1 in plan.retrieval_levels and summary_kind == SummaryViewKind.EPISODE.value
        if hierarchy_level == 2:
            return 2 in plan.retrieval_levels and summary_kind == SummaryViewKind.THEMATIC_PROFILE.value
        return False

    @staticmethod
    def _candidate_matches_scope(candidate: dict[str, Any], plan: RetrievalPlan) -> bool:
        scope = str(candidate.get("scope_canonical") or candidate.get("scope") or "")
        allowed = set(MemoryObjectRepository.canonical_retrieval_scopes(plan.scope_filter))
        persona_matches = candidate.get("user_persona_id") == plan.user_persona_id
        if not persona_matches:
            return False
        if scope == MemoryScope.USER.value:
            return MemoryScope.USER in allowed and (not plan.incognito) and plan.remember_across_chats
        if scope == MemoryScope.CHARACTER.value:
            expected_character_id = plan.character_id if plan.character_id is not None else plan.workspace_id
            return (
                MemoryScope.CHARACTER in allowed
                and (not plan.incognito)
                and plan.remember_across_chats
                and expected_character_id is not None
                and candidate.get("character_id") == expected_character_id
            )
        if scope == MemoryScope.CHAT.value:
            if CandidateSearch._privacy_sql_filters_disabled(plan):
                return MemoryScope.CHAT in allowed and (not plan.incognito) and plan.remember_across_chats
            return (
                MemoryScope.CHAT in allowed
                and candidate.get("conversation_id") == plan.conversation_id
            )
        return False

    @staticmethod
    def _scope_priority(scope_value: str, scope_filter: list[MemoryScope]) -> int:
        canonical_value = _canonical_scope_value(scope_value)
        for index, scope in enumerate(MemoryObjectRepository.canonical_retrieval_scopes(scope_filter)):
            if scope.value == canonical_value:
                return index
        return len(scope_filter)

    @staticmethod
    def _retrieval_level_clauses(plan: RetrievalPlan, alias: str = "mo") -> tuple[list[str], list[Any]]:
        _assert_safe_alias(alias)
        clauses: list[str] = []
        parameters: list[Any] = []
        if 0 in plan.retrieval_levels:
            clauses.append(
                "("
                f"{alias}.object_type != 'summary_view' "
                "OR ("
                f"{alias}.object_type = 'summary_view' "
                f"AND CAST(json_extract({alias}.payload_json, '$.hierarchy_level') AS INTEGER) = 0 "
                f"AND json_extract({alias}.payload_json, '$.summary_kind') = ?"
                ")"
                ")"
            )
            parameters.append(SummaryViewKind.CONVERSATION_CHUNK.value)
        if 1 in plan.retrieval_levels:
            clauses.append(
                "("
                f"{alias}.object_type = 'summary_view' "
                f"AND CAST(json_extract({alias}.payload_json, '$.hierarchy_level') AS INTEGER) = 1 "
                f"AND json_extract({alias}.payload_json, '$.summary_kind') = ?"
                ")"
            )
            parameters.append(SummaryViewKind.EPISODE.value)
        if 2 in plan.retrieval_levels:
            clauses.append(
                "("
                f"{alias}.object_type = 'summary_view' "
                f"AND CAST(json_extract({alias}.payload_json, '$.hierarchy_level') AS INTEGER) = 2 "
                f"AND json_extract({alias}.payload_json, '$.summary_kind') = ?"
                ")"
            )
            parameters.append(SummaryViewKind.THEMATIC_PROFILE.value)
        return clauses, parameters

    @staticmethod
    def _retrieval_level_priority(candidate: dict[str, Any], plan: RetrievalPlan) -> int:
        if candidate.get("object_type") != "summary_view":
            level = 0
        else:
            payload_json = candidate.get("payload_json") or {}
            level = int(payload_json.get("hierarchy_level", 99)) if isinstance(payload_json, dict) else 99
        try:
            return plan.retrieval_levels.index(level)
        except ValueError:
            return len(plan.retrieval_levels)

    @staticmethod
    def _updated_at_sort_key(updated_at: str) -> float:
        return datetime.fromisoformat(updated_at).timestamp()

    def _sort_embedding_candidates(
        self,
        candidates: list[dict[str, Any]],
        plan: RetrievalPlan,
    ) -> list[dict[str, Any]]:
        return sorted(
            candidates,
            key=lambda candidate: (
                self._temporal_priority(candidate, plan),
                self._retrieval_level_priority(candidate, plan),
                int(candidate.get("position_rank", 10**9)),
                -self._updated_at_sort_key(str(candidate["updated_at"])),
                str(candidate["id"]),
            ),
        )

    @staticmethod
    def _assign_position_ranks(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ranked: list[dict[str, Any]] = []
        for rank_index, candidate in enumerate(candidates, start=1):
            updated = dict(candidate)
            updated["position_rank"] = rank_index
            ranked.append(updated)
        return ranked

    @staticmethod
    def _annotate_fts_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
        annotated = dict(candidate)
        if "rank" in annotated:
            annotated["fts_rank"] = annotated["rank"]
        return annotated

    @staticmethod
    def _annotate_fts_query_match(
        candidate: dict[str, Any],
        *,
        sub_query_text: str,
        fts_query: str,
        fts_query_kind: str,
        fts_query_source: str = "planned",
        non_evidential: bool = True,
    ) -> dict[str, Any]:
        annotated = dict(candidate)
        match = {
            "subquery": sub_query_text,
            "query": fts_query,
            "kind": fts_query_kind,
            "match_mode": CandidateSearch._fts_query_match_mode(fts_query),
            "position_rank": int(candidate.get("position_rank") or 0),
        }
        if fts_query_source not in {"planned", "dynamic"}:
            match["source"] = fts_query_source
            match["non_evidential"] = bool(non_evidential)
        annotated["fts_query_matches"] = [match]
        return annotated

    @staticmethod
    def _record_fts_query_audit(
        audit: list[dict[str, Any]] | None,
        *,
        sub_query_text: str,
        fts_query: str,
        fts_query_kind: str,
        raw_rows: int,
        source: str = "planned",
        non_evidential: bool = True,
    ) -> None:
        if audit is None:
            return
        entry = {
            "subquery": sub_query_text,
            "query": fts_query,
            "kind": fts_query_kind,
            "match_mode": CandidateSearch._fts_query_match_mode(fts_query),
            "raw_rows": max(0, int(raw_rows)),
        }
        if source not in {"planned", "dynamic"}:
            entry["source"] = source
            entry["non_evidential"] = bool(non_evidential)
        audit.append(entry)

    @staticmethod
    def _fts_query_kind_at(sub_query: Any, index: int) -> str:
        kinds = list(getattr(sub_query, "fts_query_kinds", []) or [])
        if 0 <= index < len(kinds):
            return str(kinds[index] or "unknown")
        return "unknown"

    @staticmethod
    def _fts_query_match_mode(fts_query: str) -> str:
        query = str(fts_query or "").strip()
        if not query:
            return "unknown"
        if query.startswith('"') and query.endswith('"'):
            return "quoted_phrase"
        if " OR " in query:
            return "explicit_or"
        return "implicit_and"

    def _merge_channel_candidates(
        self,
        aggregated: dict[str, dict[str, Any]],
        candidates: list[dict[str, Any]],
        *,
        channel: str,
        plan: RetrievalPlan,
    ) -> None:
        for candidate in candidates:
            position_rank = candidate.get("position_rank")
            if position_rank is None:
                raise ValueError(f"Candidate {candidate.get('id')} missing position_rank for channel {channel}")

            memory_id = str(candidate["id"])
            current = aggregated.get(memory_id)
            current_was_created = current is None
            if current is None:
                current = self._initialize_channel_candidate(candidate)
                aggregated[memory_id] = current

            existing_rank = current["channel_ranks"].get(channel)
            if existing_rank is None or int(position_rank) < int(existing_rank):
                for key, value in candidate.items():
                    if key in {
                        "channel_ranks",
                        "retrieval_sources",
                        "rrf_score",
                        "rrf_score_raw",
                        "retrieval_source",
                        "fts_query_matches",
                    }:
                        continue
                    current[key] = value
                current["channel_ranks"][channel] = int(position_rank)

            if not current_was_created:
                self._merge_fts_query_matches(current, candidate)

            if channel not in current["retrieval_sources"]:
                current["retrieval_sources"].append(channel)
                current["retrieval_sources"].sort(key=_CHANNEL_ORDER.index)

            current["position_rank"] = min(
                rank
                for rank in current["channel_ranks"].values()
                if rank is not None
            )
            current["retrieval_source"] = "+".join(current["retrieval_sources"])
            raw_rrf_score, normalized_rrf_score = self._compute_rrf_scores(
                current["channel_ranks"],
                query_type=plan.query_type,
                exact_recall_mode=plan.exact_recall_mode,
                raw_context_access_mode=plan.raw_context_access_mode,
            )
            current["rrf_score_raw"] = raw_rrf_score
            current["rrf_score"] = normalized_rrf_score

    def _merge_sub_query_candidates(
        self,
        aggregated: dict[str, dict[str, Any]],
        candidates: list[dict[str, Any]],
        *,
        plan: RetrievalPlan,
        sub_query_text: str,
        total_sub_queries: int,
    ) -> None:
        for candidate in candidates:
            position_rank = candidate.get("position_rank")
            if position_rank is None:
                raise ValueError(
                    f"Candidate {candidate.get('id')} missing position_rank for sub-query {sub_query_text!r}"
                )

            memory_id = str(candidate["id"])
            current = aggregated.get(memory_id)
            current_was_created = current is None
            if current is None:
                current = dict(candidate)
                current["matched_sub_queries"] = []
                current["subquery_ranks"] = {}
                aggregated[memory_id] = current

            existing_rank = current["subquery_ranks"].get(sub_query_text)
            if existing_rank is None or int(position_rank) < int(existing_rank):
                for key, value in candidate.items():
                    if key in {
                        "matched_sub_queries",
                        "subquery_ranks",
                        "channel_ranks",
                        "retrieval_sources",
                        "retrieval_source",
                        "rrf_score",
                        "rrf_score_raw",
                        "position_rank",
                        "fts_query_matches",
                    }:
                        continue
                    current[key] = value
                current["subquery_ranks"][sub_query_text] = int(position_rank)

            if not current_was_created:
                self._merge_fts_query_matches(current, candidate)

            if sub_query_text not in current["matched_sub_queries"]:
                current["matched_sub_queries"].append(sub_query_text)

            current_sources = list(current.get("retrieval_sources", []))
            seen_sources = set(current_sources)
            for source in candidate.get("retrieval_sources", []):
                if source in seen_sources:
                    continue
                current_sources.append(source)
                seen_sources.add(source)
            current_sources.sort(key=_CHANNEL_ORDER.index)
            current["retrieval_sources"] = current_sources
            current["retrieval_source"] = "+".join(current_sources)

            current_channel_ranks = current.get("channel_ranks", {})
            candidate_channel_ranks = candidate.get("channel_ranks", {})
            if isinstance(current_channel_ranks, dict) and isinstance(candidate_channel_ranks, dict):
                for channel, rank in candidate_channel_ranks.items():
                    if rank is None:
                        continue
                    existing_channel_rank = current_channel_ranks.get(channel)
                    if existing_channel_rank is None or int(rank) < int(existing_channel_rank):
                        current_channel_ranks[channel] = int(rank)
                current["channel_ranks"] = current_channel_ranks

            current["position_rank"] = min(current["subquery_ranks"].values())
            raw_rrf_score, normalized_rrf_score = self._compute_rank_fusion_scores(
                current["subquery_ranks"].values(),
                max_lists=total_sub_queries,
                query_type=plan.query_type,
                exact_recall_mode=plan.exact_recall_mode,
                raw_context_access_mode=plan.raw_context_access_mode,
                channel_ranks=current.get("channel_ranks"),
            )
            current["rrf_score_raw"] = raw_rrf_score
            current["rrf_score"] = normalized_rrf_score

    def _initialize_channel_candidate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        channel_candidate = dict(candidate)
        channel_candidate["channel_ranks"] = {
            channel: None for channel in _CHANNEL_ORDER
        }
        channel_candidate["retrieval_sources"] = []
        channel_candidate["rrf_score"] = 0.0
        channel_candidate["rrf_score_raw"] = 0.0
        channel_candidate["retrieval_source"] = ""
        return channel_candidate

    @staticmethod
    def _merge_fts_query_matches(
        current: dict[str, Any],
        candidate: dict[str, Any],
    ) -> None:
        incoming = candidate.get("fts_query_matches")
        if not isinstance(incoming, list) or not incoming:
            return
        existing = current.get("fts_query_matches")
        if not isinstance(existing, list):
            existing = []
        seen = {
            (
                str(match.get("subquery") or ""),
                str(match.get("query") or ""),
                str(match.get("kind") or ""),
                str(match.get("match_mode") or ""),
            )
            for match in existing
            if isinstance(match, dict)
        }
        merged = list(existing)
        for match in incoming:
            if not isinstance(match, dict):
                continue
            signature = (
                str(match.get("subquery") or ""),
                str(match.get("query") or ""),
                str(match.get("kind") or ""),
                str(match.get("match_mode") or ""),
            )
            if signature in seen:
                continue
            seen.add(signature)
            merged.append(dict(match))
        current["fts_query_matches"] = merged

    def _compute_rrf_scores(
        self,
        channel_ranks: dict[str, int | None],
        *,
        query_type: str = "default",
        exact_recall_mode: bool = False,
        raw_context_access_mode: str = "normal",
    ) -> tuple[float, float]:
        raw_score = 0.0
        for channel, rank in channel_ranks.items():
            if rank is None:
                continue
            raw_score += self._channel_weight(
                channel,
                query_type=query_type,
                exact_recall_mode=exact_recall_mode,
                raw_context_access_mode=raw_context_access_mode,
            ) * (1.0 / (self._rrf_k + int(rank)))
        normalized_score = raw_score / self._max_rrf_score(
            query_type=query_type,
            exact_recall_mode=exact_recall_mode,
            raw_context_access_mode=raw_context_access_mode,
        )
        return raw_score, max(0.0, min(1.0, normalized_score))

    def _max_rrf_score(
        self,
        *,
        query_type: str = "default",
        exact_recall_mode: bool = False,
        raw_context_access_mode: str = "normal",
    ) -> float:
        # Upper bound assumes every channel contributes its best rank,
        # with the weighted raw evidence channel included so a pure raw
        # hit can normalise against the same ceiling as a full-hit.
        return sum(
            self._channel_weight(
                channel,
                query_type=query_type,
                exact_recall_mode=exact_recall_mode,
                raw_context_access_mode=raw_context_access_mode,
            ) * (1.0 / (self._rrf_k + 1))
            for channel in _CHANNEL_ORDER
        )

    def _channel_weight(
        self,
        channel: str,
        *,
        query_type: str = "default",
        exact_recall_mode: bool = False,
        raw_context_access_mode: str = "normal",
    ) -> float:
        multiplier_map = self._channel_multiplier_map(
            query_type=query_type,
            exact_recall_mode=exact_recall_mode,
            raw_context_access_mode=raw_context_access_mode,
        )
        multiplier = multiplier_map.get(channel, 1.0)
        if channel == "fact_facet":
            return self._fact_facet_retrieval_weight * multiplier
        if channel == "verbatim_evidence_search":
            return self._verbatim_evidence_search_weight * multiplier
        return multiplier

    @staticmethod
    def _channel_multiplier_map(
        *,
        query_type: str,
        exact_recall_mode: bool,
        raw_context_access_mode: str,
    ) -> dict[str, float]:
        if exact_recall_mode or raw_context_access_mode in {"artifact", "verbatim"}:
            return _EXACT_RECALL_CHANNEL_MULTIPLIERS
        return _QUERY_TYPE_CHANNEL_MULTIPLIERS.get(query_type, {})

    def _compute_rank_fusion_scores(
        self,
        ranks: Any,
        *,
        max_lists: int,
        query_type: str,
        exact_recall_mode: bool = False,
        raw_context_access_mode: str = "normal",
        channel_ranks: dict[str, int | None] | None = None,
    ) -> tuple[float, float]:
        rank_values = [int(rank) for rank in ranks if rank is not None]
        if not rank_values:
            return 0.0, 0.0
        raw_score = sum(1.0 / (self._rrf_k + rank) for rank in rank_values)
        # Channel-weight propagation: intra-sub-query fusion applies
        # per-channel weights, but final fusion collapses sub-query ranks
        # without channel info. Honor channel weights end-to-end by
        # down-weighting candidates that only ever matched through
        # below-unit channels (e.g. the verbatim_evidence_search safety-net channel).
        # A candidate that also matched a unit-weight channel keeps the
        # full score.
        candidate_channel_weight = self._candidate_channel_weight(
            channel_ranks,
            query_type=query_type,
            exact_recall_mode=exact_recall_mode,
            raw_context_access_mode=raw_context_access_mode,
        )
        raw_score *= candidate_channel_weight
        if query_type == "broad_list":
            score_list_count = max_lists
        else:
            # Outside broad-list queries, preserve the strength of the best matched
            # sub-query instead of heavily rewarding generic coverage across facets.
            score_list_count = len(rank_values)
        if score_list_count <= 0:
            return 0.0, 0.0
        max_score = sum(1.0 / (self._rrf_k + 1) for _ in range(score_list_count))
        normalized_score = raw_score / max_score
        return raw_score, max(0.0, min(1.0, normalized_score))

    def _candidate_channel_weight(
        self,
        channel_ranks: dict[str, int | None] | None,
        *,
        query_type: str = "default",
        exact_recall_mode: bool = False,
        raw_context_access_mode: str = "normal",
    ) -> float:
        """Return the effective channel weight for a merged candidate.

        The weight is the max weight among channels that actually
        contributed a rank for this candidate. A candidate seen via a
        full-weight channel (e.g. ``fts``) keeps weight 1.0 even if it
        also matched via the lighter verbatim_evidence_search channel; a candidate
        that ONLY matched via verbatim_evidence_search is down-weighted by that
        channel's weight.
        """
        if not channel_ranks:
            return 1.0
        contributing_weights = [
            self._channel_weight(
                channel,
                query_type=query_type,
                exact_recall_mode=exact_recall_mode,
                raw_context_access_mode=raw_context_access_mode,
            )
            for channel, rank in channel_ranks.items()
            if rank is not None
        ]
        if not contributing_weights:
            return 1.0
        return max(contributing_weights)

    def _sort_candidates(
        self,
        candidates: Any,
        plan: RetrievalPlan,
    ) -> list[dict[str, Any]]:
        return sorted(
            candidates,
            key=lambda candidate: (
                self._temporal_priority(candidate, plan),
                self._callback_priority(candidate, plan),
                self._retrieval_level_priority(candidate, plan),
                self._scope_priority(candidate["scope"], plan.scope_filter),
                -float(candidate.get("rrf_score", 0.0)),
                -self._updated_at_sort_key(candidate["updated_at"]),
                str(candidate["id"]),
            ),
        )

    @classmethod
    def _is_lower_position_rank(
        cls,
        candidate: dict[str, Any],
        current: dict[str, Any],
    ) -> bool:
        return (
            int(candidate["position_rank"]),
            -cls._updated_at_sort_key(str(candidate["updated_at"])),
            str(candidate["id"]),
        ) < (
            int(current["position_rank"]),
            -cls._updated_at_sort_key(str(current["updated_at"])),
            str(current["id"]),
        )

    def _temporal_priority(self, candidate: dict[str, Any], plan: RetrievalPlan) -> int:
        temporal_type = str(candidate.get("temporal_type", "unknown"))
        if plan.temporal_query_range is None:
            if temporal_type == "ephemeral" and self._is_stale_ephemeral(candidate):
                return _STALE_EPHEMERAL_PRIORITY
            return _TEMPORAL_OVERLAP_PRIORITY
        if temporal_type == "unknown":
            return _TEMPORAL_UNKNOWN_PRIORITY
        if temporal_type == "ephemeral":
            valid_from, effective_end = self._ephemeral_effective_range(
                candidate,
                reference=plan.temporal_query_range.start,
            )
            if valid_from is None or effective_end is None:
                return _TEMPORAL_NON_OVERLAP_PRIORITY
            if self._overlaps_temporal_range(valid_from, effective_end, plan):
                return _TEMPORAL_OVERLAP_PRIORITY
            return _TEMPORAL_NON_OVERLAP_PRIORITY
        valid_from = self._parse_temporal_datetime(
            candidate.get("valid_from"),
            reference=plan.temporal_query_range.start,
        )
        valid_to = self._parse_temporal_datetime(
            candidate.get("valid_to"),
            reference=plan.temporal_query_range.start,
        )
        if self._overlaps_temporal_range(valid_from, valid_to, plan):
            return _TEMPORAL_OVERLAP_PRIORITY
        return _TEMPORAL_NON_OVERLAP_PRIORITY

    @staticmethod
    def _callback_priority(candidate: dict[str, Any], plan: RetrievalPlan) -> int:
        if not plan.callback_bias:
            return 0
        return 0 if bool(candidate.get("assistant_source_match")) else 1

    async def _populate_callback_source_flags(
        self,
        candidates: list[dict[str, Any]],
        *,
        user_id: str,
    ) -> None:
        memory_ids = [str(candidate["id"]) for candidate in candidates if candidate.get("id") is not None]
        if not memory_ids:
            return
        placeholders = ", ".join("?" for _ in memory_ids)
        cursor = await self._connection.execute(
            f"""
            SELECT
                mo.id AS memory_id,
                MAX(CASE WHEN m.role = 'assistant' THEN 1 ELSE 0 END) AS assistant_source_match
            FROM memory_objects AS mo
            LEFT JOIN json_each(mo.payload_json, '$.source_message_ids') AS source_ids ON 1 = 1
            LEFT JOIN messages AS m ON m.id = source_ids.value
            WHERE mo.user_id = ?
              AND mo.id IN ({placeholders})
            GROUP BY mo.id
            """,
            (user_id, *memory_ids),
        )
        rows = await cursor.fetchall()
        assistant_matches = {
            str(row["memory_id"]): bool(row["assistant_source_match"])
            for row in rows
        }
        for candidate in candidates:
            candidate["assistant_source_match"] = assistant_matches.get(
                str(candidate["id"]),
                False,
            )

    @staticmethod
    def _parse_temporal_datetime(value: Any, *, reference: datetime | None) -> datetime | None:
        if value is None:
            return None
        parsed = datetime.fromisoformat(str(value))
        if parsed.tzinfo is None and reference is not None and reference.tzinfo is not None:
            parsed = parsed.replace(tzinfo=reference.tzinfo)
        return parsed

    @staticmethod
    def _overlaps_temporal_range(
        valid_from: datetime | None,
        valid_to: datetime | None,
        plan: RetrievalPlan,
    ) -> bool:
        if plan.temporal_query_range is None:
            return True
        return (
            (valid_from is None or valid_from <= plan.temporal_query_range.end)
            and (valid_to is None or valid_to >= plan.temporal_query_range.start)
        )

    def _fts_channel_limit(self, plan: RetrievalPlan) -> int:
        if plan.temporal_query_range is not None:
            return plan.max_candidates
        return plan.max_candidates * _CHANNEL_OVERFETCH_MULTIPLIER

    @staticmethod
    def _effective_embedding_limit(
        plan: RetrievalPlan,
        embedding_index: EmbeddingIndex,
    ) -> int:
        backend_limit = max(0, int(embedding_index.vector_limit))
        if backend_limit == 0 or plan.vector_limit == 0:
            return 0
        return min(plan.vector_limit, backend_limit)

    def _embedding_search_limit(self, vector_limit: int) -> int:
        return vector_limit * self._settings.embedding_search_overfetch_multiplier

    def _is_stale_ephemeral(self, candidate: dict[str, Any]) -> bool:
        valid_from, effective_end = self._ephemeral_effective_range(
            candidate,
            reference=self._clock.now(),
        )
        return valid_from is None or effective_end is None or effective_end < self._clock.now()

    def _ephemeral_effective_range(
        self,
        candidate: dict[str, Any],
        *,
        reference: datetime,
    ) -> tuple[datetime | None, datetime | None]:
        valid_from = self._parse_temporal_datetime(candidate.get("valid_from"), reference=reference)
        if valid_from is None:
            return None, None
        valid_to = self._parse_temporal_datetime(candidate.get("valid_to"), reference=reference)
        if valid_to is not None:
            return valid_from, valid_to
        return valid_from, valid_from + timedelta(hours=self._settings.ephemeral_scoring_hours)
