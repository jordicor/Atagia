"""Candidate search over SQLite and FTS indexes."""

from __future__ import annotations

from datetime import datetime, timedelta
import re
from typing import Any

import aiosqlite

from atagia.core.clock import Clock
from atagia.core.config import Settings
from atagia.core.artifact_repository import ArtifactRepository
from atagia.core.repositories import MessageRepository, _decode_json_columns
from atagia.core.verbatim_pin_repository import VerbatimPinRepository
from atagia.models.schemas_memory import (
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    MemoryStatus,
    RetrievalPlan,
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
# Channels participating in rank fusion. ``raw_message`` carries raw
# conversation evidence (Wave 1 batch 2, task 1-C) and is weighted
# slightly lower than curated memory channels by default.
_CHANNEL_ORDER = ("verbatim_pin", "artifact_chunk", "fts", "embedding", "consequence", "raw_message")
_RAW_MESSAGE_ID_PREFIX = "rmw_"
# Widest-to-narrowest order used to assign a synthetic scope to a raw
# conversation window. The widest scope compatible with both the plan
# and the window's owning conversation wins.
_RAW_SCOPE_BREADTH_ORDER = (
    MemoryScope.GLOBAL_USER,
    MemoryScope.ASSISTANT_MODE,
    MemoryScope.WORKSPACE,
    MemoryScope.CONVERSATION,
    MemoryScope.EPHEMERAL_SESSION,
)
_SAFE_ALIAS_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CHANNEL_OVERFETCH_MULTIPLIER = 2
_TEMPORAL_OVERLAP_PRIORITY = 0
_TEMPORAL_UNKNOWN_PRIORITY = 1
_TEMPORAL_NON_OVERLAP_PRIORITY = 2
_STALE_EPHEMERAL_PRIORITY = 1


def _assert_safe_alias(alias: str) -> None:
    if not _SAFE_ALIAS_PATTERN.match(alias):
        raise ValueError(f"Unsafe SQL alias: {alias!r}")


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
        # Wave 1 batch 2 (1-C): raw-message channel wiring.
        self._raw_message_channel_enabled = self._settings.raw_message_channel
        self._raw_message_channel_weight = float(self._settings.raw_message_rrf_weight)
        self._raw_message_channel_limit = int(self._settings.raw_message_channel_limit)
        self._raw_message_window_size = int(self._settings.raw_message_window_size)
        self._raw_message_window_overlap = int(self._settings.raw_message_window_overlap)

    async def search(
        self,
        plan: RetrievalPlan,
        user_id: str,
    ) -> list[dict[str, Any]]:
        if plan.skip_retrieval or plan.max_candidates <= 0:
            return []

        scope_clauses, scope_parameters = self._scope_clauses(
            scope_filter=plan.scope_filter,
            assistant_mode_id=plan.assistant_mode_id,
            workspace_id=plan.workspace_id,
            conversation_id=plan.conversation_id,
        )
        if not scope_clauses or not plan.status_filter or not plan.sub_query_plans:
            return []

        status_placeholders = ", ".join("?" for _ in plan.status_filter)
        scope_order_case = self._scope_order_case(plan.scope_filter)
        temporal_order_case = self._temporal_order_case(plan, alias="mo")
        retrieval_level_clauses, retrieval_level_parameters = self._retrieval_level_clauses(plan)
        fts_limit = self._fts_channel_limit(plan)
        query = """
            SELECT
                mo.*,
                bm25(memory_objects_fts) AS rank
            FROM memory_objects_fts
            JOIN memory_objects AS mo ON mo._rowid = memory_objects_fts.rowid
            WHERE mo.user_id = ?
              AND ({scope_clauses})
              AND mo.status IN ({status_placeholders})
              AND mo.privacy_level <= ?
              AND ({retrieval_level_clauses})
              AND memory_objects_fts MATCH ?
            ORDER BY {temporal_order_case}{scope_order_case}, rank ASC, mo.updated_at DESC
            LIMIT ?
        """.format(
            scope_clauses=" OR ".join(scope_clauses),
            status_placeholders=status_placeholders,
            retrieval_level_clauses=" OR ".join(retrieval_level_clauses),
            temporal_order_case=temporal_order_case,
            scope_order_case=scope_order_case,
        )

        aggregated: dict[str, dict[str, Any]] = {}
        for sub_query in plan.sub_query_plans:
            sub_query_aggregated: dict[str, dict[str, Any]] = {}
            for fts_query in sub_query.fts_queries:
                parameters = (
                    user_id,
                    *scope_parameters,
                    *(status.value for status in plan.status_filter),
                    plan.privacy_ceiling,
                    *retrieval_level_parameters,
                    fts_query,
                    *self._temporal_order_parameters(plan),
                    fts_limit,
                )
                cursor = await self._connection.execute(query, parameters)
                rows = await cursor.fetchall()
                fts_candidates = self._assign_position_ranks(
                    [
                        self._annotate_fts_candidate(decoded)
                        for decoded in (_decode_json_columns(row) for row in rows)
                        if decoded is not None
                    ]
                )
                self._merge_channel_candidates(
                    sub_query_aggregated,
                    fts_candidates,
                    channel="fts",
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
            )
            artifact_chunk_candidates = await self._search_artifact_chunks(
                plan=plan,
                user_id=user_id,
                sub_query=sub_query,
            )
            self._merge_channel_candidates(
                sub_query_aggregated,
                artifact_chunk_candidates,
                channel="artifact_chunk",
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
            )
            # Wave 1 batch 2 (1-C): raw evidence lane. Each sub-query
            # contributes conversation windows from the messages table
            # alongside the memory-object channels. Privacy filtering
            # runs at the SQL layer, so results that reach here are
            # already scoped to the current ceiling.
            raw_message_candidates = await self._search_raw_messages(
                plan=plan,
                user_id=user_id,
                sub_query=sub_query,
            )
            self._merge_channel_candidates(
                sub_query_aggregated,
                raw_message_candidates,
                channel="raw_message",
            )
            # Collapse raw windows that overlap memory-object candidates
            # already retrieved for this sub-query. Memories are more
            # focused, raw evidence is the recall safety net.
            self._dedupe_raw_messages_against_memories(sub_query_aggregated)
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

        # Cross-sub-query dedup pass: a raw window fetched under one
        # sub-query may overlap a memory object fetched under a different
        # sub-query. The per-sub-query dedup call above only sees one
        # sub-query at a time, so we repeat the pass against the fully
        # merged aggregated set before final ranking.
        self._dedupe_raw_messages_against_memories(aggregated)
        if plan.callback_bias and aggregated:
            await self._populate_callback_source_flags(
                list(aggregated.values()),
                user_id=user_id,
            )
        return self._sort_candidates(aggregated.values(), plan)[: plan.max_candidates]

    async def aggregate_retrievable_language_mix(
        self,
        *,
        user_id: str,
        scope_filter: list[MemoryScope],
        assistant_mode_id: str,
        workspace_id: str | None,
        conversation_id: str,
        privacy_ceiling: int,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        scope_clauses, scope_parameters = self._scope_clauses(
            scope_filter=scope_filter,
            assistant_mode_id=assistant_mode_id,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
        )
        if not scope_clauses:
            return []
        query = """
            SELECT
                lc.value AS language_code,
                COUNT(*) AS memory_count,
                MAX(mo.updated_at) AS last_seen_at
            FROM memory_objects AS mo
            JOIN json_each(mo.language_codes_json) AS lc
            WHERE mo.user_id = ?
              AND ({scope_clauses})
              AND mo.status = ?
              AND mo.privacy_level <= ?
              AND mo.language_codes_json IS NOT NULL
            GROUP BY lc.value
            ORDER BY memory_count DESC, last_seen_at DESC, language_code ASC
            LIMIT ?
        """.format(
            scope_clauses=" OR ".join(scope_clauses),
        )
        cursor = await self._connection.execute(
            query,
            (
                user_id,
                *scope_parameters,
                MemoryStatus.ACTIVE.value,
                privacy_ceiling,
                limit,
            ),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def search_by_embedding(
        self,
        *,
        query_text: str,
        user_id: str,
        plan: RetrievalPlan,
        embedding_index: EmbeddingIndex,
    ) -> list[dict[str, Any]]:
        if embedding_index.vector_limit == 0 or plan.vector_limit == 0:
            return []
        matches = await embedding_index.search(
            query_text,
            user_id,
            self._embedding_channel_limit(plan),
        )
        if not matches:
            return []

        candidates: list[dict[str, Any]] = []
        for match in matches:
            row = await self._load_memory_object(match.memory_id, user_id)
            if row is None or not self._matches_plan_filters(row, plan):
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
            return candidates[: plan.vector_limit]
        return self._sort_embedding_candidates(self._assign_position_ranks(candidates), plan)[: plan.vector_limit]

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
                privacy_ceiling=plan.privacy_ceiling,
                scope_filter=plan.scope_filter,
                assistant_mode_id=plan.assistant_mode_id,
                workspace_id=plan.workspace_id,
                conversation_id=plan.conversation_id,
                limit=self._fts_channel_limit(plan),
                as_of=self._clock.now().isoformat(),
            )
            for row in rows:
                pin_candidate = self._build_verbatim_pin_candidate(
                    row,
                    sub_query_text=sub_query.text,
                )
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
    ) -> list[dict[str, Any]]:
        if not (plan.exact_recall_mode or plan.raw_context_access_mode == "artifact"):
            return []
        if not sub_query.fts_queries:
            return []

        aggregated: dict[str, dict[str, Any]] = {}
        queries_to_try = list(sub_query.fts_queries)
        if plan.exact_recall_mode:
            queries_to_try = queries_to_try[:1]

        for fts_query in queries_to_try:
            rows = await self._artifact_repository.search_artifact_chunks(
                user_id=user_id,
                query=fts_query,
                privacy_ceiling=plan.privacy_ceiling,
                scope_filter=plan.scope_filter,
                assistant_mode_id=plan.assistant_mode_id,
                workspace_id=plan.workspace_id,
                conversation_id=plan.conversation_id,
                limit=self._fts_channel_limit(plan),
            )
            for row in rows:
                artifact_candidate = self._build_artifact_chunk_candidate(
                    row,
                    plan=plan,
                    sub_query_text=sub_query.text,
                )
                if artifact_candidate.get("scope") is None:
                    continue
                memory_id = str(artifact_candidate["id"])
                existing = aggregated.get(memory_id)
                if existing is None or float(artifact_candidate.get("rank", 0.0)) < float(
                    existing.get("rank", 0.0)
                ):
                    aggregated[memory_id] = artifact_candidate
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
        chain_relevance_clauses = ["cc.conversation_id = ?"]
        chain_relevance_parameters: list[Any] = [plan.conversation_id]
        if plan.workspace_id is not None:
            chain_relevance_clauses.append("cc.workspace_id = ?")
            chain_relevance_parameters.append(plan.workspace_id)
        status_placeholders = ", ".join("?" for _ in plan.status_filter)
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
                COALESCE(tendency.object_type, outcome.object_type) AS object_type,
                COALESCE(tendency.scope, outcome.scope) AS scope,
                COALESCE(tendency.canonical_text, outcome.canonical_text) AS canonical_text,
                COALESCE(tendency.extraction_hash, outcome.extraction_hash) AS extraction_hash,
                COALESCE(tendency.payload_json, outcome.payload_json) AS payload_json,
                COALESCE(tendency.source_kind, outcome.source_kind) AS source_kind,
                COALESCE(tendency.confidence, outcome.confidence) AS confidence,
                COALESCE(tendency.stability, outcome.stability) AS stability,
                COALESCE(tendency.vitality, outcome.vitality) AS vitality,
                COALESCE(tendency.maya_score, outcome.maya_score) AS maya_score,
                COALESCE(tendency.privacy_level, outcome.privacy_level) AS privacy_level,
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
                bm25(memory_objects_fts) AS rank
            FROM memory_objects_fts
            JOIN memory_objects AS matched ON matched._rowid = memory_objects_fts.rowid
            JOIN consequence_chains AS cc ON matched.id = cc.{match_column}
            LEFT JOIN memory_objects AS tendency ON tendency.id = cc.tendency_belief_id
            JOIN memory_objects AS action ON action.id = cc.action_memory_id
            JOIN memory_objects AS outcome ON outcome.id = cc.outcome_memory_id
            WHERE cc.user_id = ?
              AND cc.status = 'active'
              AND matched.user_id = ?
              AND COALESCE(tendency.user_id, outcome.user_id) = ?
              AND COALESCE(tendency.status, outcome.status) IN ({status_placeholders})
              AND COALESCE(tendency.privacy_level, outcome.privacy_level) <= ?
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
            candidate_scope_clauses=" OR ".join(candidate_scope_clauses),
            chain_relevance_clauses=" OR ".join(chain_relevance_clauses),
            temporal_order_case=temporal_order_case,
            scope_order_case=scope_order_case,
        )
        parameters = (
            user_id,
            user_id,
            user_id,
            *(status.value for status in plan.status_filter),
            plan.privacy_ceiling,
            *candidate_scope_parameters,
            *chain_relevance_parameters,
            fts_query,
            *self._temporal_order_parameters(plan),
            self._fts_channel_limit(plan),
        )
        cursor = await self._connection.execute(query, parameters)
        rows = await cursor.fetchall()
        return [
            decoded
            for decoded in (_decode_json_columns(row) for row in rows)
            if decoded is not None
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
        candidate = {
            "_rowid": pin_row.get("_rowid"),
            "id": str(pin_row["id"]),
            "user_id": str(pin_row["user_id"]),
            "workspace_id": pin_row.get("workspace_id"),
            "conversation_id": pin_row.get("conversation_id"),
            "assistant_mode_id": pin_row.get("assistant_mode_id"),
            "scope": str(pin_row["scope"]),
            "target_kind": str(pin_row["target_kind"]),
            "target_id": str(pin_row["target_id"]),
            "target_span_start": pin_row.get("target_span_start"),
            "target_span_end": pin_row.get("target_span_end"),
            "canonical_text": str(pin_row["canonical_text"]),
            "index_text": str(pin_row["index_text"]),
            "privacy_level": int(pin_row.get("privacy_level", 0)),
            "status": str(pin_row["status"]),
            "reason": pin_row.get("reason"),
            "created_by": str(pin_row["created_by"]),
            "created_at": str(pin_row["created_at"]),
            "updated_at": str(pin_row["updated_at"]),
            "expires_at": pin_row.get("expires_at"),
            "deleted_at": pin_row.get("deleted_at"),
            "payload_json": {
                **payload,
                "verbatim_pin_id": str(pin_row["id"]),
                "verbatim_pin_target_kind": str(pin_row["target_kind"]),
                "verbatim_pin_target_id": str(pin_row["target_id"]),
            },
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
            "channel_ranks": {"verbatim_pin": None, "fts": None, "embedding": None, "consequence": None, "raw_message": None},
            "matched_sub_queries": [sub_query_text],
            "is_verbatim_pin": True,
            "retrieval_level": 0,
        }
        candidate["channel_ranks"]["verbatim_pin"] = 1
        return candidate

    @staticmethod
    def _scope_clauses(
        *,
        scope_filter: list[MemoryScope],
        assistant_mode_id: str,
        workspace_id: str | None,
        conversation_id: str,
        alias: str = "mo",
    ) -> tuple[list[str], list[Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        for scope in scope_filter:
            if scope is MemoryScope.GLOBAL_USER:
                clauses.append(f"({alias}.scope = 'global_user')")
            elif scope is MemoryScope.ASSISTANT_MODE:
                clauses.append(f"({alias}.scope = 'assistant_mode' AND {alias}.assistant_mode_id = ?)")
                parameters.append(assistant_mode_id)
            elif scope is MemoryScope.WORKSPACE and workspace_id is not None:
                clauses.append(
                    f"({alias}.scope = 'workspace' AND {alias}.assistant_mode_id = ? AND {alias}.workspace_id = ?)"
                )
                parameters.extend([assistant_mode_id, workspace_id])
            elif scope is MemoryScope.CONVERSATION:
                clauses.append(
                    f"({alias}.scope = 'conversation' AND {alias}.assistant_mode_id = ? AND {alias}.conversation_id = ?)"
                )
                parameters.extend([assistant_mode_id, conversation_id])
            elif scope is MemoryScope.EPHEMERAL_SESSION:
                clauses.append(
                    f"({alias}.scope = 'ephemeral_session' AND {alias}.assistant_mode_id = ? AND {alias}.conversation_id = ?)"
                )
                parameters.extend([assistant_mode_id, conversation_id])
        return clauses, parameters

    @staticmethod
    def _scope_order_case(scope_filter: list[MemoryScope], alias: str = "mo") -> str:
        clauses = [
            f"WHEN {alias}.scope = '{scope.value}' THEN {index}"
            for index, scope in enumerate(scope_filter)
        ]
        return "CASE " + " ".join(clauses) + f" ELSE {len(scope_filter)} END"

    def _temporal_order_parameters(self, plan: RetrievalPlan) -> tuple[str, ...]:
        if plan.temporal_query_range is None:
            return ()
        query_start_minus_horizon = (
            plan.temporal_query_range.start
            - timedelta(hours=self._settings.ephemeral_scoring_hours)
        ).isoformat()
        return (
            plan.temporal_query_range.end.isoformat(),
            plan.temporal_query_range.start.isoformat(),
            query_start_minus_horizon,
            plan.temporal_query_range.end.isoformat(),
            plan.temporal_query_range.start.isoformat(),
        )

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
        scope_expr = f"COALESCE({primary_alias}.scope, {fallback_alias}.scope)"
        assistant_mode_expr = f"COALESCE({primary_alias}.assistant_mode_id, {fallback_alias}.assistant_mode_id)"
        workspace_expr = f"COALESCE({primary_alias}.workspace_id, {fallback_alias}.workspace_id)"
        conversation_expr = f"COALESCE({primary_alias}.conversation_id, {fallback_alias}.conversation_id)"

        clauses: list[str] = []
        parameters: list[Any] = []
        for scope in plan.scope_filter:
            if scope is MemoryScope.GLOBAL_USER:
                clauses.append(f"({scope_expr} = 'global_user')")
            elif scope is MemoryScope.ASSISTANT_MODE:
                clauses.append(f"({scope_expr} = 'assistant_mode' AND {assistant_mode_expr} = ?)")
                parameters.append(plan.assistant_mode_id)
            elif scope is MemoryScope.WORKSPACE and plan.workspace_id is not None:
                clauses.append(
                    f"({scope_expr} = 'workspace' AND {assistant_mode_expr} = ? AND {workspace_expr} = ?)"
                )
                parameters.extend([plan.assistant_mode_id, plan.workspace_id])
            elif scope is MemoryScope.CONVERSATION:
                clauses.append(
                    f"({scope_expr} = 'conversation' AND {assistant_mode_expr} = ? AND {conversation_expr} = ?)"
                )
                parameters.extend([plan.assistant_mode_id, plan.conversation_id])
            elif scope is MemoryScope.EPHEMERAL_SESSION:
                clauses.append(
                    f"({scope_expr} = 'ephemeral_session' AND {assistant_mode_expr} = ? AND {conversation_expr} = ?)"
                )
                parameters.extend([plan.assistant_mode_id, plan.conversation_id])
        return clauses, parameters

    @staticmethod
    def _coalesced_scope_order_case(
        scope_filter: list[MemoryScope],
        *,
        primary_alias: str,
        fallback_alias: str,
    ) -> str:
        scope_expr = f"COALESCE({primary_alias}.scope, {fallback_alias}.scope)"
        clauses = [
            f"WHEN {scope_expr} = '{scope.value}' THEN {index}"
            for index, scope in enumerate(scope_filter)
        ]
        return "CASE " + " ".join(clauses) + f" ELSE {len(scope_filter)} END"

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

    async def _load_memory_object(self, memory_id: str, user_id: str) -> dict[str, Any] | None:
        cursor = await self._connection.execute(
            """
            SELECT *
            FROM memory_objects
            WHERE id = ?
              AND user_id = ?
            """,
            (memory_id, user_id),
        )
        row = await cursor.fetchone()
        return _decode_json_columns(row)

    # ------------------------------------------------------------------
    # Wave 1 batch 2 (1-C): raw evidence channel
    # ------------------------------------------------------------------

    async def _search_raw_messages(
        self,
        *,
        plan: RetrievalPlan,
        user_id: str,
        sub_query: Any,
    ) -> list[dict[str, Any]]:
        """Return ranked conversation windows for a single sub-query.

        Steps:
        1. Run a privacy-filtered FTS query on messages.
        2. Materialize a ``raw_message_window_size`` conversation window
           around each match. Overlap across neighbouring matches is
           collapsed by keeping the best-ranked seed per window slot.
        3. Shape the window as a pseudo-candidate compatible with the
           downstream fusion / filtering pipeline.
        """
        if not self._raw_message_channel_enabled:
            return []
        if self._raw_message_channel_limit <= 0:
            return []

        channel_scope = self._pick_raw_message_scope(plan.scope_filter)
        if channel_scope is None:
            return []

        # Raw evidence is strongest on the precise FTS variant. Planner
        # order already goes from most precise to broader rewrites.
        #
        # For exact-recall questions we still prefer that strongest
        # query first, but a multilingual bridge can make the precise
        # rewrite too strict for verbatim transcript search. In that
        # case we fall back to broader rewrites only if the stricter
        # query found no windows.
        fts_queries = list(sub_query.fts_queries or [])
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
        for fts_query in queries_to_try:
            rows = await self._message_repository.search_messages_with_privacy(
                user_id=user_id,
                query=fts_query,
                privacy_ceiling=plan.privacy_ceiling,
                limit=self._raw_message_channel_limit,
                allow_conversation_id=plan.conversation_id,
            )
            for row in rows:
                window_candidate = await self._build_raw_message_window_candidate(
                    seed_row=row,
                    plan=plan,
                    user_id=user_id,
                    channel_scope=channel_scope,
                )
                if window_candidate is None:
                    continue
                window_id = str(window_candidate["id"])
                existing = aggregated.get(window_id)
                if existing is None or float(window_candidate.get("fts_rank", 0.0)) < float(
                    existing.get("fts_rank", 0.0)
                ):
                    aggregated[window_id] = window_candidate
            if plan.exact_recall_mode and aggregated:
                break

        ordered = sorted(
            aggregated.values(),
            key=lambda candidate: (
                float(candidate.get("fts_rank", 0.0)),
                str(candidate["id"]),
            ),
        )
        return self._assign_position_ranks(ordered[: self._raw_message_channel_limit])

    async def _build_raw_message_window_candidate(
        self,
        *,
        seed_row: dict[str, Any],
        plan: RetrievalPlan,
        user_id: str,
        channel_scope: MemoryScope,
    ) -> dict[str, Any] | None:
        """Shape a seed message row into a conversation-window candidate."""
        conversation_id = str(seed_row["conversation_id"])
        seed_seq = int(seed_row["seq"])
        window_messages = await self._message_repository.fetch_message_window(
            conversation_id=conversation_id,
            user_id=user_id,
            center_seq=seed_seq,
            window_size=self._raw_message_window_size,
        )
        if not window_messages:
            return None
        conversation_mode_id = str(seed_row.get("conversation_assistant_mode_id") or "")
        # Scope mapping guards cross-mode and cross-conversation leakage
        # for the downstream scope filter. The SQL layer already checked
        # the mode privacy ceiling.
        resolved_scope = self._resolve_raw_window_scope(
            channel_scope=channel_scope,
            plan=plan,
            conversation_mode_id=conversation_mode_id,
            conversation_id=conversation_id,
        )
        if resolved_scope is None:
            return None

        start_seq = int(window_messages[0]["seq"])
        end_seq = int(window_messages[-1]["seq"])
        window_id = f"{_RAW_MESSAGE_ID_PREFIX}{conversation_id}_{start_seq}_{end_seq}"
        canonical_text = self._format_raw_window_text(window_messages)
        occurred_at = (
            window_messages[-1].get("occurred_at")
            or window_messages[-1].get("created_at")
        )
        source_message_ids = [str(message["id"]) for message in window_messages]
        payload: dict[str, Any] = {
            "source_message_ids": source_message_ids,
            "source_kind_variant": "conversation_window",
            "window_start_seq": start_seq,
            "window_end_seq": end_seq,
            "window_size": len(window_messages),
            "source_window_start_occurred_at": str(
                window_messages[0].get("occurred_at")
                or window_messages[0].get("created_at")
                or ""
            ),
            "source_window_end_occurred_at": str(
                window_messages[-1].get("occurred_at")
                or window_messages[-1].get("created_at")
                or ""
            ),
        }
        fts_rank = float(seed_row.get("rank") or 0.0)
        return {
            "_rowid": None,
            "id": window_id,
            "user_id": user_id,
            "workspace_id": plan.workspace_id,
            "conversation_id": conversation_id,
            "assistant_mode_id": conversation_mode_id,
            "object_type": MemoryObjectType.EVIDENCE.value,
            "scope": resolved_scope.value,
            "canonical_text": canonical_text,
            "payload_json": payload,
            "source_kind": MemorySourceKind.VERBATIM.value,
            "confidence": 1.0,
            "stability": 0.5,
            "vitality": 0.0,
            "maya_score": 0.0,
            "privacy_level": 0,
            "temporal_type": "unknown",
            "valid_from": None,
            "valid_to": None,
            "status": MemoryStatus.ACTIVE.value,
            "created_at": str(seed_row.get("created_at")),
            "updated_at": str(occurred_at or seed_row.get("created_at")),
            "rank": fts_rank,
            "fts_rank": fts_rank,
            "retrieval_level": 0,
            "is_raw_message_window": True,
            "raw_window_conversation_id": conversation_id,
            "raw_window_start_seq": start_seq,
            "raw_window_end_seq": end_seq,
            "raw_window_message_ids": source_message_ids,
        }

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
            "object_type": MemoryObjectType.EVIDENCE.value,
            "scope": CandidateSearch._scope_for_artifact_row(row, plan),
            "canonical_text": canonical_text,
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
            },
            "source_kind": MemorySourceKind.VERBATIM.value
            if preserve_verbatim
            else MemorySourceKind.EXTRACTED.value,
            "confidence": 1.0 if preserve_verbatim else 0.9,
            "stability": 0.5,
            "vitality": 0.0,
            "maya_score": 0.0,
            "privacy_level": int(row.get("privacy_level", 0)),
            "temporal_type": "unknown",
            "valid_from": None,
            "valid_to": None,
            "status": "active",
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
            "embedding": None,
            "consequence": None,
            "raw_message": None,
        }
        candidate["retrieval_source"] = "artifact_chunk"
        candidate["matched_sub_queries"] = [sub_query_text]
        return candidate

    @staticmethod
    def _format_raw_window_text(messages: list[dict[str, Any]]) -> str:
        """Format a conversation window as a compact transcript block."""
        lines: list[str] = []
        for message in messages:
            role = str(message.get("role") or "user")
            text = str(message.get("text") or "").strip()
            if not text:
                continue
            lines.append(f"[{role}] {text}")
        return "\n".join(lines)

    @staticmethod
    def _scope_for_artifact_row(row: dict[str, Any], plan: RetrievalPlan) -> str | None:
        conversation_mode_id = str(row.get("conversation_assistant_mode_id") or "")
        workspace_id = row.get("workspace_id")
        conversation_id = str(row.get("conversation_id") or "")
        if (
            MemoryScope.CONVERSATION in plan.scope_filter
            and conversation_id == plan.conversation_id
            and conversation_mode_id == plan.assistant_mode_id
        ):
            return MemoryScope.CONVERSATION.value
        if (
            MemoryScope.EPHEMERAL_SESSION in plan.scope_filter
            and conversation_id == plan.conversation_id
            and conversation_mode_id == plan.assistant_mode_id
        ):
            return MemoryScope.EPHEMERAL_SESSION.value
        if (
            MemoryScope.WORKSPACE in plan.scope_filter
            and workspace_id == plan.workspace_id
            and conversation_mode_id == plan.assistant_mode_id
        ):
            return MemoryScope.WORKSPACE.value
        if (
            MemoryScope.ASSISTANT_MODE in plan.scope_filter
            and conversation_mode_id == plan.assistant_mode_id
        ):
            return MemoryScope.ASSISTANT_MODE.value
        if (
            MemoryScope.GLOBAL_USER in plan.scope_filter
            and workspace_id is None
            and not conversation_id
        ):
            return MemoryScope.GLOBAL_USER.value
        return None

    @staticmethod
    def _pick_raw_message_scope(scope_filter: list[MemoryScope]) -> MemoryScope | None:
        """Pick the widest plan-allowed scope for raw evidence windows."""
        allowed = set(scope_filter)
        for scope in _RAW_SCOPE_BREADTH_ORDER:
            if scope in allowed:
                return scope
        return None

    @staticmethod
    def _resolve_raw_window_scope(
        *,
        channel_scope: MemoryScope,
        plan: RetrievalPlan,
        conversation_mode_id: str,
        conversation_id: str,
    ) -> MemoryScope | None:
        """Map a raw window to a scope consistent with plan filtering.

        The returned scope must satisfy ``_candidate_matches_scope`` so
        that windows survive the post-fusion filter. Narrower scopes
        require a matching conversation or mode identifier; wider scopes
        (``global_user``) accept any conversation that already cleared
        the SQL-layer privacy check.
        """
        if channel_scope is MemoryScope.GLOBAL_USER:
            return MemoryScope.GLOBAL_USER
        if channel_scope is MemoryScope.ASSISTANT_MODE:
            if conversation_mode_id == plan.assistant_mode_id:
                return MemoryScope.ASSISTANT_MODE
            return None
        if channel_scope is MemoryScope.WORKSPACE:
            if conversation_mode_id != plan.assistant_mode_id:
                return None
            # Workspace scope requires a workspace; raw windows do not
            # carry one without extra joins, so only accept when the
            # window's conversation is the active one.
            if conversation_id == plan.conversation_id:
                return MemoryScope.WORKSPACE
            return None
        if channel_scope is MemoryScope.CONVERSATION:
            if (
                conversation_mode_id == plan.assistant_mode_id
                and conversation_id == plan.conversation_id
            ):
                return MemoryScope.CONVERSATION
            return None
        if channel_scope is MemoryScope.EPHEMERAL_SESSION:
            if (
                conversation_mode_id == plan.assistant_mode_id
                and conversation_id == plan.conversation_id
            ):
                return MemoryScope.EPHEMERAL_SESSION
            return None
        return None

    @staticmethod
    def _dedupe_raw_messages_against_memories(
        aggregated: dict[str, dict[str, Any]],
    ) -> None:
        """Drop raw-window candidates already covered by a memory object.

        A raw window is considered covered when any of its source
        message ids appears in ``payload_json.source_message_ids`` of a
        non-raw candidate that also scored for the same sub-query.
        """
        covered_message_ids: set[str] = set()
        for candidate in aggregated.values():
            if candidate.get("is_raw_message_window"):
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
            if not candidate.get("is_raw_message_window"):
                continue
            window_message_ids = {
                str(message_id)
                for message_id in candidate.get("raw_window_message_ids") or []
            }
            if window_message_ids & covered_message_ids:
                to_drop.append(window_id)
        for window_id in to_drop:
            aggregated.pop(window_id, None)

    @classmethod
    def _matches_plan_filters(cls, candidate: dict[str, Any], plan: RetrievalPlan) -> bool:
        if int(candidate.get("privacy_level", 0)) > plan.privacy_ceiling:
            return False
        if candidate.get("status") not in {status.value for status in plan.status_filter}:
            return False
        if not cls._candidate_matches_retrieval_levels(candidate, plan):
            return False
        return cls._candidate_matches_scope(candidate, plan)

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
        scope = str(candidate.get("scope", ""))
        if scope == MemoryScope.GLOBAL_USER.value:
            return MemoryScope.GLOBAL_USER in plan.scope_filter
        if scope == MemoryScope.ASSISTANT_MODE.value:
            return (
                MemoryScope.ASSISTANT_MODE in plan.scope_filter
                and candidate.get("assistant_mode_id") == plan.assistant_mode_id
            )
        if scope == MemoryScope.WORKSPACE.value:
            return (
                MemoryScope.WORKSPACE in plan.scope_filter
                and candidate.get("assistant_mode_id") == plan.assistant_mode_id
                and candidate.get("workspace_id") == plan.workspace_id
            )
        if scope == MemoryScope.CONVERSATION.value:
            return (
                MemoryScope.CONVERSATION in plan.scope_filter
                and candidate.get("assistant_mode_id") == plan.assistant_mode_id
                and candidate.get("conversation_id") == plan.conversation_id
            )
        if scope == MemoryScope.EPHEMERAL_SESSION.value:
            return (
                MemoryScope.EPHEMERAL_SESSION in plan.scope_filter
                and candidate.get("assistant_mode_id") == plan.assistant_mode_id
                and candidate.get("conversation_id") == plan.conversation_id
            )
        return False

    @staticmethod
    def _scope_priority(scope_value: str, scope_filter: list[MemoryScope]) -> int:
        for index, scope in enumerate(scope_filter):
            if scope.value == scope_value:
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

    def _merge_channel_candidates(
        self,
        aggregated: dict[str, dict[str, Any]],
        candidates: list[dict[str, Any]],
        *,
        channel: str,
    ) -> None:
        for candidate in candidates:
            position_rank = candidate.get("position_rank")
            if position_rank is None:
                raise ValueError(f"Candidate {candidate.get('id')} missing position_rank for channel {channel}")

            memory_id = str(candidate["id"])
            current = aggregated.get(memory_id)
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
                    }:
                        continue
                    current[key] = value
                current["channel_ranks"][channel] = int(position_rank)

            if channel not in current["retrieval_sources"]:
                current["retrieval_sources"].append(channel)
                current["retrieval_sources"].sort(key=_CHANNEL_ORDER.index)

            current["position_rank"] = min(
                rank
                for rank in current["channel_ranks"].values()
                if rank is not None
            )
            current["retrieval_source"] = "+".join(current["retrieval_sources"])
            raw_rrf_score, normalized_rrf_score = self._compute_rrf_scores(current["channel_ranks"])
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
                    }:
                        continue
                    current[key] = value
                current["subquery_ranks"][sub_query_text] = int(position_rank)

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

    def _compute_rrf_scores(self, channel_ranks: dict[str, int | None]) -> tuple[float, float]:
        raw_score = 0.0
        for channel, rank in channel_ranks.items():
            if rank is None:
                continue
            raw_score += self._channel_weight(channel) * (1.0 / (self._rrf_k + int(rank)))
        normalized_score = raw_score / self._max_rrf_score()
        return raw_score, max(0.0, min(1.0, normalized_score))

    def _max_rrf_score(self) -> float:
        # Upper bound assumes every channel contributes its best rank,
        # with the weighted raw evidence channel included so a pure raw
        # hit can normalise against the same ceiling as a full-hit.
        return sum(
            self._channel_weight(channel) * (1.0 / (self._rrf_k + 1))
            for channel in _CHANNEL_ORDER
        )

    def _channel_weight(self, channel: str) -> float:
        if channel == "raw_message":
            return self._raw_message_channel_weight
        return 1.0

    def _compute_rank_fusion_scores(
        self,
        ranks: Any,
        *,
        max_lists: int,
        query_type: str,
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
        # below-unit channels (e.g. the raw_message safety-net channel).
        # A candidate that also matched a unit-weight channel keeps the
        # full score.
        candidate_channel_weight = self._candidate_channel_weight(channel_ranks)
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
    ) -> float:
        """Return the effective channel weight for a merged candidate.

        The weight is the max weight among channels that actually
        contributed a rank for this candidate. A candidate seen via a
        full-weight channel (e.g. ``fts``) keeps weight 1.0 even if it
        also matched via the lighter raw_message channel; a candidate
        that ONLY matched via raw_message is down-weighted by that
        channel's weight.
        """
        if not channel_ranks:
            return 1.0
        contributing_weights = [
            self._channel_weight(channel)
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
    def _embedding_channel_limit(plan: RetrievalPlan) -> int:
        return plan.vector_limit * _CHANNEL_OVERFETCH_MULTIPLIER

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
