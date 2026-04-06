"""Candidate search over SQLite and FTS indexes."""

from __future__ import annotations

from datetime import datetime
import re
from typing import Any

import aiosqlite

from atagia.core.clock import Clock
from atagia.core.config import Settings
from atagia.core.repositories import _decode_json_columns
from atagia.models.schemas_memory import MemoryScope, RetrievalPlan, SummaryViewKind
from atagia.services.embeddings import EmbeddingIndex, NoneBackend

_ALLOWED_CONSEQUENCE_MATCH_COLUMNS = frozenset(
    {
        "action_memory_id",
        "outcome_memory_id",
        "tendency_belief_id",
    }
)
_CHANNEL_ORDER = ("fts", "embedding", "consequence")
_SAFE_ALIAS_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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

    async def search(
        self,
        plan: RetrievalPlan,
        user_id: str,
        *,
        query_text: str | None = None,
    ) -> list[dict[str, Any]]:
        if plan.skip_retrieval or plan.max_candidates <= 0:
            return []

        scope_clauses, scope_parameters = self._scope_clauses(plan)
        if not scope_clauses or not plan.status_filter:
            return []

        status_placeholders = ", ".join("?" for _ in plan.status_filter)
        scope_order_case = self._scope_order_case(plan.scope_filter)
        temporal_order_case = self._temporal_order_case(plan, alias="mo")
        retrieval_level_clauses, retrieval_level_parameters = self._retrieval_level_clauses(plan)
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
        if plan.fts_queries:
            for fts_query in plan.fts_queries:
                parameters = (
                    user_id,
                    *scope_parameters,
                    *(status.value for status in plan.status_filter),
                    plan.privacy_ceiling,
                    *retrieval_level_parameters,
                    fts_query,
                    *self._temporal_order_parameters(plan),
                    plan.max_candidates,
                )
                cursor = await self._connection.execute(query, parameters)
                rows = await cursor.fetchall()
                fts_candidates = self._assign_position_ranks(
                    [
                        decoded
                        for decoded in (_decode_json_columns(row) for row in rows)
                        if decoded is not None
                    ]
                )
                self._merge_channel_candidates(aggregated, fts_candidates, channel="fts")
        if plan.fts_queries and self._needs_consequence_chain_search(plan):
            consequence_candidates = await self._search_consequence_chains(
                plan=plan,
                user_id=user_id,
                fts_query=plan.fts_queries[0],
            )
            self._merge_channel_candidates(
                aggregated,
                consequence_candidates,
                channel="consequence",
            )
        if query_text:
            embedding_candidates = await self.search_by_embedding(
                query_text=query_text,
                user_id=user_id,
                plan=plan,
                embedding_index=self._embedding_index,
            )
            self._merge_channel_candidates(
                aggregated,
                embedding_candidates,
                channel="embedding",
            )

        candidates = sorted(
            aggregated.values(),
            key=lambda candidate: (
                self._temporal_priority(candidate, plan),
                self._retrieval_level_priority(candidate, plan),
                self._scope_priority(candidate["scope"], plan.scope_filter),
                -float(candidate.get("rrf_score", 0.0)),
                -self._updated_at_sort_key(candidate["updated_at"]),
                str(candidate["id"]),
            ),
        )
        return candidates[: plan.max_candidates]

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
        matches = await embedding_index.search(query_text, user_id, plan.vector_limit)
        if not matches:
            return []

        candidates: list[dict[str, Any]] = []
        for match in matches:
            row = await self._load_memory_object(match.memory_id, user_id)
            if row is None or not self._matches_plan_filters(row, plan):
                continue
            row["similarity_score"] = match.score
            row["rank"] = max(0.0, 1.0 - float(match.score))
            if match.position_rank is not None:
                row["embedding_position_rank"] = int(match.position_rank)
                row["position_rank"] = int(match.position_rank)
            candidates.append(row)
        candidates = self._sort_embedding_candidates(candidates, plan)
        if candidates and all(candidate.get("position_rank") is not None for candidate in candidates):
            return candidates[: plan.vector_limit]
        return self._sort_embedding_candidates(self._assign_position_ranks(candidates), plan)[: plan.vector_limit]

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
        )[: plan.max_candidates]

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
            plan.max_candidates,
        )
        cursor = await self._connection.execute(query, parameters)
        rows = await cursor.fetchall()
        return [
            decoded
            for decoded in (_decode_json_columns(row) for row in rows)
            if decoded is not None
        ]

    @staticmethod
    def _scope_clauses(plan: RetrievalPlan, alias: str = "mo") -> tuple[list[str], list[Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        for scope in plan.scope_filter:
            if scope is MemoryScope.GLOBAL_USER:
                clauses.append(f"({alias}.scope = 'global_user')")
            elif scope is MemoryScope.ASSISTANT_MODE:
                clauses.append(f"({alias}.scope = 'assistant_mode' AND {alias}.assistant_mode_id = ?)")
                parameters.append(plan.assistant_mode_id)
            elif scope is MemoryScope.WORKSPACE and plan.workspace_id is not None:
                clauses.append(
                    f"({alias}.scope = 'workspace' AND {alias}.assistant_mode_id = ? AND {alias}.workspace_id = ?)"
                )
                parameters.extend([plan.assistant_mode_id, plan.workspace_id])
            elif scope is MemoryScope.CONVERSATION:
                clauses.append(
                    f"({alias}.scope = 'conversation' AND {alias}.assistant_mode_id = ? AND {alias}.conversation_id = ?)"
                )
                parameters.extend([plan.assistant_mode_id, plan.conversation_id])
            elif scope is MemoryScope.EPHEMERAL_SESSION:
                clauses.append(
                    f"({alias}.scope = 'ephemeral_session' AND {alias}.assistant_mode_id = ? AND {alias}.conversation_id = ?)"
                )
                parameters.extend([plan.assistant_mode_id, plan.conversation_id])
        return clauses, parameters

    @staticmethod
    def _scope_order_case(scope_filter: list[MemoryScope], alias: str = "mo") -> str:
        clauses = [
            f"WHEN {alias}.scope = '{scope.value}' THEN {index}"
            for index, scope in enumerate(scope_filter)
        ]
        return "CASE " + " ".join(clauses) + f" ELSE {len(scope_filter)} END"

    @staticmethod
    def _temporal_order_parameters(plan: RetrievalPlan) -> tuple[str, ...]:
        if plan.temporal_query_range is None:
            return ()
        return (
            plan.temporal_query_range.end.isoformat(),
            plan.temporal_query_range.start.isoformat(),
        )

    @staticmethod
    def _temporal_order_case(plan: RetrievalPlan, *, alias: str) -> str:
        if plan.temporal_query_range is None:
            return ""
        return (
            "CASE "
            f"WHEN {alias}.temporal_type = 'unknown' THEN 1 "
            f"WHEN (({alias}.valid_from IS NULL OR {alias}.valid_from <= ?) "
            f"AND ({alias}.valid_to IS NULL OR {alias}.valid_to >= ?)) THEN 0 "
            "ELSE 2 END, "
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

    @staticmethod
    def _coalesced_temporal_order_case(
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
            f"WHEN {temporal_type_expr} = 'unknown' THEN 1 "
            f"WHEN (({valid_from_expr} IS NULL OR {valid_from_expr} <= ?) "
            f"AND ({valid_to_expr} IS NULL OR {valid_to_expr} >= ?)) THEN 0 "
            "ELSE 2 END, "
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

    @classmethod
    def _sort_embedding_candidates(
        cls,
        candidates: list[dict[str, Any]],
        plan: RetrievalPlan,
    ) -> list[dict[str, Any]]:
        return sorted(
            candidates,
            key=lambda candidate: (
                cls._temporal_priority(candidate, plan),
                cls._retrieval_level_priority(candidate, plan),
                int(candidate.get("position_rank", 10**9)),
                -cls._updated_at_sort_key(str(candidate["updated_at"])),
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
                    if key in {"channel_ranks", "retrieval_sources", "rrf_score"}:
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
        raw_score = sum(
            1.0 / (self._rrf_k + int(rank))
            for rank in channel_ranks.values()
            if rank is not None
        )
        normalized_score = raw_score / self._max_rrf_score()
        return raw_score, max(0.0, min(1.0, normalized_score))

    def _max_rrf_score(self) -> float:
        return sum(1.0 / (self._rrf_k + 1) for _ in _CHANNEL_ORDER)

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

    @classmethod
    def _temporal_priority(cls, candidate: dict[str, Any], plan: RetrievalPlan) -> int:
        if plan.temporal_query_range is None:
            return 0
        temporal_type = str(candidate.get("temporal_type", "unknown"))
        if temporal_type == "unknown":
            return 1
        valid_from = cls._parse_temporal_datetime(candidate.get("valid_from"), plan)
        valid_to = cls._parse_temporal_datetime(candidate.get("valid_to"), plan)
        if cls._overlaps_temporal_range(valid_from, valid_to, plan):
            return 0
        return 2

    @staticmethod
    def _parse_temporal_datetime(value: Any, plan: RetrievalPlan) -> datetime | None:
        if value is None:
            return None
        parsed = datetime.fromisoformat(str(value))
        if (
            parsed.tzinfo is None
            and plan.temporal_query_range is not None
            and plan.temporal_query_range.start.tzinfo is not None
        ):
            parsed = parsed.replace(tzinfo=plan.temporal_query_range.start.tzinfo)
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
