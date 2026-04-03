"""Candidate search over SQLite and FTS indexes."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import aiosqlite

from atagia.core.clock import Clock
from atagia.core.repositories import _decode_json_columns
from atagia.services.embeddings import EmbeddingIndex, NoneBackend
from atagia.models.schemas_memory import MemoryScope, NeedTrigger, RetrievalPlan

_ALLOWED_CONSEQUENCE_MATCH_COLUMNS = frozenset(
    {
        "action_memory_id",
        "outcome_memory_id",
        "tendency_belief_id",
    }
)


class CandidateSearch:
    """Executes retrieval plans against SQLite FTS indexes."""

    def __init__(
        self,
        connection: aiosqlite.Connection,
        clock: Clock,
        embedding_index: EmbeddingIndex | None = None,
    ) -> None:
        self._connection = connection
        self._clock = clock
        self._embedding_index = embedding_index or NoneBackend()

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
              AND memory_objects_fts MATCH ?
            ORDER BY {scope_order_case}, rank ASC, mo.updated_at DESC
            LIMIT ?
        """.format(
            scope_clauses=" OR ".join(scope_clauses),
            status_placeholders=status_placeholders,
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
                    fts_query,
                    plan.max_candidates,
                )
                cursor = await self._connection.execute(query, parameters)
                rows = await cursor.fetchall()
                for row in rows:
                    decoded = _decode_json_columns(row)
                    if decoded is None:
                        continue
                    current = aggregated.get(decoded["id"])
                    if current is None or self._is_better_candidate(decoded, current, plan.scope_filter):
                        aggregated[decoded["id"]] = decoded
                if self._needs_consequence_chain_search(plan):
                    consequence_candidates = await self._search_consequence_chains(
                        plan=plan,
                        user_id=user_id,
                        fts_query=fts_query,
                    )
                    for candidate in consequence_candidates:
                        current = aggregated.get(candidate["id"])
                        if current is None or self._is_better_candidate(candidate, current, plan.scope_filter):
                            aggregated[candidate["id"]] = candidate
        if query_text:
            embedding_candidates = await self.search_by_embedding(
                query_text=query_text,
                user_id=user_id,
                plan=plan,
                embedding_index=self._embedding_index,
            )
            for candidate in embedding_candidates:
                current = aggregated.get(candidate["id"])
                if current is None or self._is_better_candidate(candidate, current, plan.scope_filter):
                    aggregated[candidate["id"]] = candidate

        candidates = sorted(
            aggregated.values(),
            key=lambda candidate: (
                self._scope_priority(candidate["scope"], plan.scope_filter),
                -self._candidate_strength(candidate),
                -self._updated_at_sort_key(candidate["updated_at"]),
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
            row["retrieval_source"] = "embedding"
            candidates.append(row)
        return candidates

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
            for row in rows:
                current = aggregated.get(row["id"])
                if current is None or self._is_better_candidate(row, current, plan.scope_filter):
                    aggregated[row["id"]] = row
        return sorted(
            aggregated.values(),
            key=lambda candidate: (
                self._scope_priority(candidate["scope"], plan.scope_filter),
                float(candidate["rank"]),
                -self._updated_at_sort_key(candidate["updated_at"]),
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
                {scope_order_case},
                rank ASC,
                cc.confidence DESC,
                COALESCE(tendency.updated_at, outcome.updated_at) DESC
            LIMIT ?
        """.format(
            match_column=match_column,
            status_placeholders=status_placeholders,
            candidate_scope_clauses=" OR ".join(candidate_scope_clauses),
            chain_relevance_clauses=" OR ".join(chain_relevance_clauses),
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
    def _needs_consequence_chain_search(plan: RetrievalPlan) -> bool:
        return (
            NeedTrigger.FOLLOW_UP_FAILURE in plan.need_driven_boosts
            or NeedTrigger.LOOP in plan.need_driven_boosts
        )

    @classmethod
    def _is_better_candidate(
        cls,
        candidate: dict[str, Any],
        current: dict[str, Any],
        scope_filter: list[MemoryScope],
    ) -> bool:
        return (
            cls._scope_priority(candidate["scope"], scope_filter),
            -cls._candidate_strength(candidate),
            -cls._updated_at_sort_key(candidate["updated_at"]),
        ) < (
            cls._scope_priority(current["scope"], scope_filter),
            -cls._candidate_strength(current),
            -cls._updated_at_sort_key(current["updated_at"]),
        )

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
        return cls._candidate_matches_scope(candidate, plan)

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
    def _candidate_strength(candidate: dict[str, Any]) -> float:
        similarity_score = candidate.get("similarity_score")
        if similarity_score is not None:
            return float(similarity_score)
        rank = candidate.get("rank")
        if rank is None:
            return 0.0
        return 1.0 / (1.0 + abs(float(rank)))

    @staticmethod
    def _scope_priority(scope_value: str, scope_filter: list[MemoryScope]) -> int:
        for index, scope in enumerate(scope_filter):
            if scope.value == scope_value:
                return index
        return len(scope_filter)

    @staticmethod
    def _updated_at_sort_key(updated_at: str) -> float:
        return datetime.fromisoformat(updated_at).timestamp()
