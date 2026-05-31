"""Build durable prepared initial-context packages from canonical SQLite state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

import aiosqlite

from atagia.core import json_utils
from atagia.core.clock import Clock
from atagia.core.contract_repository import ContractDimensionRepository
from atagia.core.initial_context_package_repository import InitialContextPackageRepository
from atagia.core.repositories import (
    BaseRepository,
    MemoryObjectRepository,
    MessageRepository,
    conversation_visibility_clause,
)
from atagia.memory.context_composer import ContextComposer
from atagia.memory.embodiment_policy import embodiment_visibility_sql_clause_for_context
from atagia.memory.intimacy_boundary_policy import memory_object_intimacy_sql_clause
from atagia.memory.mind_policy import (
    annotate_overseer_grants_for_rows,
    mind_visibility_sql_clause_for_context,
)
from atagia.memory.policy_manifest import ResolvedRetrievalPolicy
from atagia.memory.realm_policy import (
    APPLICABLE_REALM_BRIDGE_MODES,
    annotate_realm_bridge_modes_for_rows,
    realm_visibility_sql_clause_for_context,
)
from atagia.memory.space_policy import space_visibility_sql_clause_for_context
from atagia.models.schemas_initial_context_package import (
    InitialContextPackageBlocks,
    InitialContextPackageBuildStatus,
    InitialContextPackageDiagnostics,
    InitialContextPackageKind,
    InitialContextPackageProfileItem,
    InitialContextPackageRecord,
)
from atagia.models.schemas_memory import (
    MemoryObjectType,
    MemoryScope,
    MemoryStatus,
    OperationalProfileSnapshot,
    SummaryViewKind,
)
from atagia.services.chat_support import (
    filter_topic_working_set_snapshot,
    format_chunk_summary,
    message_text_for_context,
    render_topic_working_set_block,
)
from atagia.services.initial_context_package_signatures import (
    build_initial_context_package_coordinate_signature,
    build_initial_context_package_policy_signature,
    build_initial_context_package_source_fingerprint,
)
from atagia.services.initial_context_package_keys import (
    build_initial_context_package_key,
    initial_context_package_subject,
)
from atagia.services.initial_context_package_curator import InitialContextPackageCurator
from atagia.services.prompt_authority import PromptAuthorityContext

INITIAL_CONTEXT_PACKAGE_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class InitialContextPackageBuildBudget:
    """Deterministic limits for package materialization."""

    package_budget_tokens: int = 2200
    profile_block_budget_tokens: int = 700
    current_state_block_budget_tokens: int = 350
    summary_block_budget_tokens: int = 450
    topic_block_budget_tokens: int = 450
    curated_block_budget_tokens: int = 450
    max_profile_items: int = 12
    max_profile_candidates: int = 36
    max_ambiguous_profile_candidates: int = 6
    min_ambiguous_profile_items: int = 2
    max_historical_profile_candidates: int = 6
    min_historical_profile_items: int = 2
    max_historical_profile_age_days: int = 45
    ambiguous_tension_threshold: float = 0.75
    historical_tension_threshold: float = 0.55
    max_curated_items: int = 8
    max_state_rows: int = 6
    max_summary_chunks: int = 2
    max_recent_seed_messages: int = 6
    max_profile_text_chars: int = 360
    max_state_value_chars: int = 260
    max_recent_seed_chars: int = 600


class InitialContextPackageBuilder(BaseRepository):
    """Materialize prompt-ready baseline and conversation packages.

    The builder is intended for background refresh jobs. Optional LLM curation
    is source-grounded and stored inside the same signed package row; prompt
    reads remain cheap and never call an LLM.
    """

    def __init__(
        self,
        connection: aiosqlite.Connection,
        clock: Clock,
        *,
        budget: InitialContextPackageBuildBudget | None = None,
        curator: InitialContextPackageCurator | None = None,
        curate_recent_verbatim_seed: bool = True,
    ) -> None:
        super().__init__(connection, clock)
        self._budget = budget or InitialContextPackageBuildBudget()
        self._curator = curator
        self._curate_recent_verbatim_seed = curate_recent_verbatim_seed
        self._package_repository = InitialContextPackageRepository(connection, clock)
        self._contract_repository = ContractDimensionRepository(connection, clock)
        self._message_repository = MessageRepository(connection, clock)

    async def build_baseline_package(
        self,
        *,
        user_id: str,
        resolved_policy: ResolvedRetrievalPolicy,
        workspace_id: str | None = None,
        assistant_mode_id: str | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        active_presence_id: str | None = None,
        active_space_id: str | None = None,
        active_space_boundary_mode: str | None = None,
        active_mind_id: str | None = None,
        mind_topology: str | None = None,
        active_embodiment_id: str | None = None,
        active_realm_id: str | None = None,
        incognito: bool = False,
        remember_across_chats: bool | None = None,
        remember_across_devices: bool | None = None,
        privacy_enforcement: str = "enforce",
        authority_context: PromptAuthorityContext | None = None,
        operational_profile: OperationalProfileSnapshot | None = None,
        commit: bool = True,
    ) -> InitialContextPackageRecord:
        """Build a query-independent baseline package for an active context."""

        user_preferences = await self._fetch_user_preferences(user_id)
        resolved_remember_chats = self._resolve_user_bool(
            remember_across_chats,
            user_preferences.get("remember_across_chats"),
            default=True,
        )
        resolved_remember_devices = self._resolve_user_bool(
            remember_across_devices,
            user_preferences.get("remember_across_devices"),
            default=True,
        )
        return await self._build_package(
            package_kind=InitialContextPackageKind.BASELINE,
            user_id=user_id,
            conversation_id=None,
            conversation=None,
            resolved_policy=resolved_policy,
            workspace_id=workspace_id,
            assistant_mode_id=assistant_mode_id or resolved_policy.profile_id.value,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            active_presence_id=active_presence_id,
            active_space_id=active_space_id,
            active_space_boundary_mode=active_space_boundary_mode,
            active_mind_id=active_mind_id,
            mind_topology=mind_topology,
            active_embodiment_id=active_embodiment_id,
            active_realm_id=active_realm_id,
            incognito=incognito,
            remember_across_chats=resolved_remember_chats,
            remember_across_devices=resolved_remember_devices,
            privacy_enforcement=privacy_enforcement,
            authority_context=authority_context,
            operational_profile=operational_profile,
            commit=commit,
        )

    async def build_conversation_package(
        self,
        *,
        user_id: str,
        conversation_id: str,
        resolved_policy: ResolvedRetrievalPolicy,
        conversation: Mapping[str, Any] | None = None,
        privacy_enforcement: str = "enforce",
        authority_context: PromptAuthorityContext | None = None,
        operational_profile: OperationalProfileSnapshot | None = None,
        remember_across_chats: bool | None = None,
        remember_across_devices: bool | None = None,
        commit: bool = True,
    ) -> InitialContextPackageRecord:
        """Build a package with bounded same-chat orientation."""

        conversation_row = (
            dict(conversation)
            if conversation is not None
            else await self._fetch_conversation(user_id=user_id, conversation_id=conversation_id)
        )
        if conversation_row is None:
            raise ValueError("conversation_id must belong to user_id")
        if str(conversation_row.get("user_id")) != user_id:
            raise ValueError("conversation must belong to user_id")
        if str(conversation_row.get("id")) != conversation_id:
            raise ValueError("conversation_id must match conversation")

        user_preferences = await self._fetch_user_preferences(user_id)
        resolved_remember_chats = self._resolve_user_bool(
            remember_across_chats,
            user_preferences.get("remember_across_chats"),
            default=True,
        )
        resolved_remember_devices = self._resolve_user_bool(
            remember_across_devices,
            user_preferences.get("remember_across_devices"),
            default=True,
        )
        incognito = self._coerce_bool(
            conversation_row.get("incognito", conversation_row.get("isolated_mode")),
        )
        return await self._build_package(
            package_kind=InitialContextPackageKind.CONVERSATION,
            user_id=user_id,
            conversation_id=conversation_id,
            conversation=conversation_row,
            resolved_policy=resolved_policy,
            workspace_id=self._optional_text(conversation_row.get("workspace_id")),
            assistant_mode_id=(
                self._optional_text(conversation_row.get("assistant_mode_id"))
                or resolved_policy.profile_id.value
            ),
            user_persona_id=self._optional_text(conversation_row.get("user_persona_id")),
            platform_id=self._optional_text(conversation_row.get("platform_id")),
            character_id=self._optional_text(conversation_row.get("character_id")),
            active_presence_id=self._optional_text(
                conversation_row.get("active_presence_id")
            ),
            active_space_id=self._optional_text(conversation_row.get("active_space_id")),
            active_space_boundary_mode=None,
            active_mind_id=self._optional_text(conversation_row.get("active_mind_id")),
            mind_topology=self._optional_text(conversation_row.get("mind_topology")),
            active_embodiment_id=self._optional_text(
                conversation_row.get("active_embodiment_id")
            ),
            active_realm_id=self._optional_text(conversation_row.get("active_realm_id")),
            incognito=incognito,
            remember_across_chats=resolved_remember_chats,
            remember_across_devices=resolved_remember_devices,
            privacy_enforcement=privacy_enforcement,
            authority_context=authority_context,
            operational_profile=operational_profile,
            commit=commit,
        )

    async def _build_package(
        self,
        *,
        package_kind: InitialContextPackageKind,
        user_id: str,
        conversation_id: str | None,
        conversation: Mapping[str, Any] | None,
        resolved_policy: ResolvedRetrievalPolicy,
        workspace_id: str | None,
        assistant_mode_id: str,
        user_persona_id: str | None,
        platform_id: str | None,
        character_id: str | None,
        active_presence_id: str | None,
        active_space_id: str | None,
        active_space_boundary_mode: str | None,
        active_mind_id: str | None,
        mind_topology: str | None,
        active_embodiment_id: str | None,
        active_realm_id: str | None,
        incognito: bool,
        remember_across_chats: bool,
        remember_across_devices: bool,
        privacy_enforcement: str,
        authority_context: PromptAuthorityContext | None,
        operational_profile: OperationalProfileSnapshot | None,
        commit: bool,
    ) -> InitialContextPackageRecord:
        retrieval_profile_id = resolved_policy.profile_id.value
        policy_signature = build_initial_context_package_policy_signature(
            resolved_policy,
            privacy_enforcement=privacy_enforcement,
            authority_context=authority_context,
            operational_profile=operational_profile,
        )
        coordinate_signature = await build_initial_context_package_coordinate_signature(
            self._connection,
            user_id=user_id,
            retrieval_profile_id=retrieval_profile_id,
            conversation_id=conversation_id,
            conversation=conversation,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            workspace_id=workspace_id,
            assistant_mode_id=assistant_mode_id,
            active_presence_id=active_presence_id,
            active_space_id=active_space_id,
            active_mind_id=active_mind_id,
            mind_topology=mind_topology,
            active_embodiment_id=active_embodiment_id,
            active_realm_id=active_realm_id,
            incognito=incognito,
            now=self._clock.now(),
        )
        resolved_space_boundary_mode = (
            active_space_boundary_mode
            or self._space_boundary_mode_from_signature(
                coordinate_signature.markers_json
            )
        )
        source_fingerprint = await build_initial_context_package_source_fingerprint(
            self._connection,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        contract_block, contract_refs = await self._build_contract_block(
            user_id=user_id,
            assistant_mode_id=assistant_mode_id,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
            remember_across_chats=remember_across_chats,
            remember_across_devices=remember_across_devices,
            resolved_policy=resolved_policy,
            active_space_id=active_space_id,
            active_space_boundary_mode=resolved_space_boundary_mode,
            active_mind_id=active_mind_id,
            mind_topology=mind_topology,
            active_embodiment_id=active_embodiment_id,
            active_realm_id=active_realm_id,
        )
        profile_items, profile_candidate_count = await self._build_profile_items(
            user_id=user_id,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
            remember_across_chats=remember_across_chats,
            remember_across_devices=remember_across_devices,
            resolved_policy=resolved_policy,
            active_space_id=active_space_id,
            active_space_boundary_mode=resolved_space_boundary_mode,
            active_mind_id=active_mind_id,
            mind_topology=mind_topology,
            active_embodiment_id=active_embodiment_id,
            active_realm_id=active_realm_id,
        )
        profile_items_before_budget = len(profile_items)
        profile_block, profile_items, dropped_profile_for_budget = self._render_profile_block(
            profile_items
        )
        dropped_profile_for_limit = max(0, profile_candidate_count - profile_items_before_budget)
        current_state_block, state_refs = await self._build_current_state_block(
            user_id=user_id,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
            remember_across_chats=remember_across_chats,
            remember_across_devices=remember_across_devices,
            resolved_policy=resolved_policy,
            active_space_id=active_space_id,
            active_space_boundary_mode=resolved_space_boundary_mode,
            active_mind_id=active_mind_id,
            mind_topology=mind_topology,
            active_embodiment_id=active_embodiment_id,
            active_realm_id=active_realm_id,
        )
        coordinate_context_block = self._render_coordinate_context_block(
            coordinate_signature.markers_json
        )
        conversation_summary_block, summary_refs = await self._build_summary_block(
            user_id=user_id,
            conversation_id=conversation_id,
            resolved_policy=resolved_policy,
        )
        working_topic_block, topic_refs = await self._build_topic_block(
            user_id=user_id,
            conversation_id=conversation_id,
            resolved_policy=resolved_policy,
        )
        recent_verbatim_seed = await self._build_recent_verbatim_seed(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        curated_items, curated_block, curation_warnings = await self._build_curated_orientation(
            user_id=user_id,
            package_kind=package_kind.value,
            retrieval_profile_id=retrieval_profile_id,
            coordinate_complete=coordinate_signature.complete,
            profile_items=profile_items,
            current_state_block=current_state_block,
            state_refs=state_refs,
            conversation_summary_block=conversation_summary_block,
            summary_refs=summary_refs,
            working_topic_block=working_topic_block,
            topic_refs=topic_refs,
            recent_verbatim_seed=recent_verbatim_seed,
        )

        blocks = InitialContextPackageBlocks(
            contract_block=contract_block,
            curated_orientation_block=curated_block,
            prepared_memory_profile_block=profile_block,
            current_state_block=current_state_block,
            coordinate_context_block=coordinate_context_block,
            conversation_summary_block=conversation_summary_block,
            working_topic_block=working_topic_block,
            recent_verbatim_seed=recent_verbatim_seed,
            empty_markers=self._empty_markers(
                profile_items=profile_items,
                curated_items=curated_items,
                current_state_block=current_state_block,
                conversation_summary_block=conversation_summary_block,
                working_topic_block=working_topic_block,
                recent_verbatim_seed=recent_verbatim_seed,
                conversation_id=conversation_id,
            ),
            source_counts=self._source_counts(
                profile_items=profile_items,
                curated_items=curated_items,
                contract_refs=contract_refs,
                state_refs=state_refs,
                summary_refs=summary_refs,
                topic_refs=topic_refs,
                recent_verbatim_seed=recent_verbatim_seed,
            ),
            curated_items=curated_items,
            profile_items=profile_items,
        )
        blocks, extra_dropped, extra_curated_dropped, budget_warnings = (
            self._enforce_package_budget(blocks)
        )
        source_refs_json = {
            "contract": contract_refs,
            "curated_orientation": [
                ref
                for item in blocks.curated_items
                for ref in item.source_refs
            ],
            "profile_items": [
                ref
                for item in blocks.profile_items
                for ref in item.source_refs
            ],
            "current_state": state_refs if blocks.current_state_block else [],
            "conversation_summary": (
                summary_refs if blocks.conversation_summary_block else []
            ),
            "working_topic": topic_refs if blocks.working_topic_block else [],
            "recent_verbatim_seed": [
                {
                    "source_kind": "message",
                    "message_id": seed.get("message_id"),
                    "conversation_id": seed.get("conversation_id"),
                    "seq": seed.get("seq"),
                }
                for seed in blocks.recent_verbatim_seed
            ],
        }
        diagnostics = InitialContextPackageDiagnostics(
            package_tokens_estimate=self._estimate_blocks_tokens(blocks),
            source_counts=blocks.source_counts,
            selected_profile_items=len(blocks.profile_items),
            dropped_profile_items=(
                dropped_profile_for_limit + dropped_profile_for_budget + extra_dropped
            ),
            selected_curated_items=len(blocks.curated_items),
            dropped_curated_items=extra_curated_dropped,
            warnings=[
                *(
                    ["coordinate_signature_incomplete"]
                    if not coordinate_signature.complete
                    else []
                ),
                *curation_warnings,
                *budget_warnings,
            ],
        )
        key = build_initial_context_package_key(
            version=INITIAL_CONTEXT_PACKAGE_SCHEMA_VERSION,
            package_kind=package_kind,
            user_id=user_id,
            conversation_id=conversation_id,
            retrieval_profile_id=retrieval_profile_id,
            subject_json=initial_context_package_subject(
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id,
                workspace_id=workspace_id,
                assistant_mode_id=assistant_mode_id,
                mode=(
                    str(conversation.get("mode"))
                    if conversation is not None and conversation.get("mode") is not None
                    else None
                ),
            ),
            policy_signature=policy_signature,
            coordinate_signature=coordinate_signature,
            operational_profile=operational_profile,
        )
        return await self._package_repository.upsert_package(
            package_kind=package_kind,
            version=INITIAL_CONTEXT_PACKAGE_SCHEMA_VERSION,
            user_id=user_id,
            conversation_id=conversation_id,
            retrieval_profile_id=retrieval_profile_id,
            key_json=key,
            policy_signature_json=policy_signature,
            coordinate_signature_json=coordinate_signature,
            source_fingerprint_json=source_fingerprint,
            blocks_json=blocks,
            source_refs_json=source_refs_json,
            diagnostics_json=diagnostics,
            build_status=InitialContextPackageBuildStatus.ACTIVE,
            commit=commit,
        )

    async def _build_contract_block(
        self,
        *,
        user_id: str,
        assistant_mode_id: str,
        workspace_id: str | None,
        conversation_id: str | None,
        user_persona_id: str | None,
        platform_id: str | None,
        character_id: str | None,
        incognito: bool,
        remember_across_chats: bool,
        remember_across_devices: bool,
        resolved_policy: ResolvedRetrievalPolicy,
        active_space_id: str | None,
        active_space_boundary_mode: str | None,
        active_mind_id: str | None,
        mind_topology: str | None,
        active_embodiment_id: str | None,
        active_realm_id: str | None,
    ) -> tuple[str, list[dict[str, Any]]]:
        rows = await self._contract_repository.list_for_context(
            user_id=user_id,
            assistant_mode_id=assistant_mode_id,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
            remember_across_chats=remember_across_chats,
            remember_across_devices=remember_across_devices,
            allow_private_sensitivity=resolved_policy.allow_private_sensitivity,
            active_space_id=active_space_id,
            active_space_boundary_mode=active_space_boundary_mode,
            active_mind_id=active_mind_id,
            mind_topology=mind_topology,
            active_embodiment_id=active_embodiment_id,
            active_realm_id=active_realm_id,
        )
        merged: dict[str, tuple[int, str, dict[str, Any], dict[str, Any]]] = {}
        for row in rows:
            dimension = str(row["dimension_name"])
            value_json = dict(row.get("value_json") or {})
            row_realm_id = self._optional_text(row.get("realm_id"))
            if active_realm_id is not None and row_realm_id is not None and row_realm_id != active_realm_id:
                value_json.setdefault(
                    "realm",
                    {
                        "active_realm_id": row_realm_id,
                        "active_request_realm_id": active_realm_id,
                        "cross_realm_mode": "applicable",
                    },
                )
            current = merged.get(dimension)
            candidate = (
                self._scope_rank(str(row.get("scope") or "")),
                str(row.get("updated_at") or ""),
                value_json,
                row,
            )
            if current is None or (candidate[0], candidate[1]) > (current[0], current[1]):
                merged[dimension] = candidate

        current_contract = {
            dimension: value
            for dimension, (_, _, value, _) in merged.items()
        }
        for dimension in await self._contract_repository.get_mode_contract_dimensions_priority(
            assistant_mode_id
        ):
            current_contract.setdefault(
                dimension,
                {"label": "default", "source": "manifest_default"},
            )
        refs = [
            {
                "source_kind": "contract_dimension",
                "contract_dimension_id": row.get("id"),
                "source_memory_id": row.get("source_memory_id"),
                "dimension_name": row.get("dimension_name"),
                "scope": row.get("scope"),
                "updated_at": row.get("updated_at"),
            }
            for _, _, _, row in merged.values()
        ]
        return (
            ContextComposer.render_contract_block(
                current_contract,
                resolved_policy,
            ),
            refs,
        )

    async def _build_profile_items(
        self,
        *,
        user_id: str,
        workspace_id: str | None,
        conversation_id: str | None,
        user_persona_id: str | None,
        platform_id: str | None,
        character_id: str | None,
        incognito: bool,
        remember_across_chats: bool,
        remember_across_devices: bool,
        resolved_policy: ResolvedRetrievalPolicy,
        active_space_id: str | None,
        active_space_boundary_mode: str | None,
        active_mind_id: str | None,
        mind_topology: str | None,
        active_embodiment_id: str | None,
        active_realm_id: str | None,
    ) -> tuple[list[InitialContextPackageProfileItem], int]:
        rows = await self._fetch_visible_memory_rows(
            user_id=user_id,
            object_types=[MemoryObjectType.BELIEF],
            statuses=[MemoryStatus.ACTIVE],
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
            remember_across_chats=remember_across_chats,
            remember_across_devices=remember_across_devices,
            resolved_policy=resolved_policy,
            active_space_id=active_space_id,
            active_space_boundary_mode=active_space_boundary_mode,
            active_mind_id=active_mind_id,
            mind_topology=mind_topology,
            active_embodiment_id=active_embodiment_id,
            active_realm_id=active_realm_id,
            limit=self._budget.max_profile_candidates,
        )
        ambiguous_rows = await self._fetch_visible_memory_rows(
            user_id=user_id,
            object_types=[MemoryObjectType.BELIEF],
            statuses=[MemoryStatus.REVIEW_REQUIRED],
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
            remember_across_chats=remember_across_chats,
            remember_across_devices=remember_across_devices,
            resolved_policy=resolved_policy,
            active_space_id=active_space_id,
            active_space_boundary_mode=active_space_boundary_mode,
            active_mind_id=active_mind_id,
            mind_topology=mind_topology,
            active_embodiment_id=active_embodiment_id,
            active_realm_id=active_realm_id,
            limit=self._budget.max_ambiguous_profile_candidates,
        )
        historical_rows = await self._fetch_visible_memory_rows(
            user_id=user_id,
            object_types=[MemoryObjectType.BELIEF],
            statuses=[MemoryStatus.SUPERSEDED],
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
            remember_across_chats=remember_across_chats,
            remember_across_devices=remember_across_devices,
            resolved_policy=resolved_policy,
            active_space_id=active_space_id,
            active_space_boundary_mode=active_space_boundary_mode,
            active_mind_id=active_mind_id,
            mind_topology=mind_topology,
            active_embodiment_id=active_embodiment_id,
            active_realm_id=active_realm_id,
            limit=self._budget.max_historical_profile_candidates,
        )
        historical_rows = [
            row for row in historical_rows if self._historical_row_is_relevant(row)
        ]
        rows = self._merge_profile_rows(rows, ambiguous_rows, historical_rows)
        refs_by_memory_id = await self._memory_source_refs_by_id(user_id, rows)
        items: list[InitialContextPackageProfileItem] = []
        for row in self._select_profile_rows(rows):
            memory_id = str(row["id"])
            text = self._compact_text(
                str(row.get("canonical_text") or ""),
                self._budget.max_profile_text_chars,
            )
            if not text:
                continue
            items.append(
                InitialContextPackageProfileItem(
                    item_id=f"memory:{memory_id}",
                    text=text,
                    reason_category=self._profile_reason_category(row),
                    source_refs=refs_by_memory_id.get(memory_id)
                    or [self._base_memory_ref(row)],
                    scope_json=self._scope_metadata(row),
                    coordinate_visibility_json=self._coordinate_visibility_metadata(row),
                    freshness_json=self._freshness_metadata(row),
                    status=self._profile_item_status(row),
                    salience=self._salience(row),
                )
            )
        return items, len(rows)

    def _select_profile_rows(
        self,
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if len(rows) <= self._budget.max_profile_items:
            return rows
        max_items = self._budget.max_profile_items
        ambiguous_target = min(
            self._budget.min_ambiguous_profile_items,
            self._budget.max_ambiguous_profile_candidates,
            max_items,
        )
        historical_target = min(
            self._budget.min_historical_profile_items,
            self._budget.max_historical_profile_candidates,
            max_items,
        )
        if ambiguous_target <= 0 and historical_target <= 0:
            return rows[:max_items]
        row_order = {
            str(row.get("id") or ""): index
            for index, row in enumerate(rows)
            if str(row.get("id") or "")
        }
        selected = list(rows[:max_items])
        selected_ids = {str(row.get("id") or "") for row in selected}
        selected = self._reserve_profile_status_rows(
            rows,
            selected=selected,
            selected_ids=selected_ids,
            statuses={"ambiguous"},
            target_count=ambiguous_target,
            start_index=max_items,
        )
        selected_ids = {str(row.get("id") or "") for row in selected}
        selected = self._reserve_profile_status_rows(
            rows,
            selected=selected,
            selected_ids=selected_ids,
            statuses={"historical", "superseded"},
            target_count=historical_target,
            start_index=max_items,
        )
        return sorted(
            selected,
            key=lambda row: row_order.get(str(row.get("id") or ""), max_items),
        )

    def _reserve_profile_status_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        selected: list[dict[str, Any]],
        selected_ids: set[str],
        statuses: set[str],
        target_count: int,
        start_index: int,
    ) -> list[dict[str, Any]]:
        if target_count <= 0:
            return selected
        selected_count = sum(
            1 for row in selected if self._profile_item_status(row) in statuses
        )
        for row in rows[start_index:]:
            if selected_count >= target_count:
                break
            memory_id = str(row.get("id") or "")
            if (
                not memory_id
                or memory_id in selected_ids
                or self._profile_item_status(row) not in statuses
            ):
                continue
            replacement_index = next(
                (
                    index
                    for index in range(len(selected) - 1, -1, -1)
                    if self._profile_item_status(selected[index]) == "current"
                ),
                None,
            )
            if replacement_index is None:
                break
            replaced_id = str(selected[replacement_index].get("id") or "")
            selected_ids.discard(replaced_id)
            selected[replacement_index] = row
            selected_ids.add(memory_id)
            selected_count += 1
        return selected

    def _render_profile_block(
        self,
        items: list[InitialContextPackageProfileItem],
    ) -> tuple[str, list[InitialContextPackageProfileItem], int]:
        selected = list(items)
        dropped = 0
        while selected:
            block = self._profile_block_text(selected)
            if self._estimate(block) <= self._budget.profile_block_budget_tokens:
                return block, selected, dropped
            selected.pop()
            dropped += 1
        return "", [], dropped

    def _profile_block_text(self, items: list[InitialContextPackageProfileItem]) -> str:
        if not items:
            return ""
        lines = [
            "[Prepared Memory Profile]",
            (
                "Use as stable orientation only; direct recent transcript and "
                "query-specific evidence outrank this block."
            ),
        ]
        for item in items:
            refs = ", ".join(
                str(ref.get("memory_id") or ref.get("source_id") or ref.get("source_kind"))
                for ref in item.source_refs[:2]
            )
            scope = item.scope_json.get("scope_canonical") or item.scope_json.get("scope")
            freshness = item.freshness_json.get("updated_at") or item.freshness_json.get(
                "created_at"
            )
            suffix_parts = [f"source: {refs}"]
            if scope:
                suffix_parts.append(f"scope: {scope}")
            if freshness:
                suffix_parts.append(f"updated: {freshness}")
            status_prefix = f"[{item.status}] " if item.status != "current" else ""
            lines.append(f"- {status_prefix}{item.text} ({'; '.join(suffix_parts)})")
        return "\n".join(lines)

    async def _build_curated_orientation(
        self,
        *,
        user_id: str,
        package_kind: str,
        retrieval_profile_id: str,
        coordinate_complete: bool,
        profile_items: list[InitialContextPackageProfileItem],
        current_state_block: str,
        state_refs: list[dict[str, Any]],
        conversation_summary_block: str,
        summary_refs: list[dict[str, Any]],
        working_topic_block: str,
        topic_refs: list[dict[str, Any]],
        recent_verbatim_seed: list[dict[str, Any]],
    ) -> tuple[list[InitialContextPackageProfileItem], str, list[str]]:
        if self._curator is None:
            return [], "", []
        result = await self._curator.curate(
            user_id=user_id,
            package_kind=package_kind,
            retrieval_profile_id=retrieval_profile_id,
            coordinate_complete=coordinate_complete,
            profile_items=profile_items,
            current_state_block=current_state_block,
            current_state_refs=state_refs,
            conversation_summary_block=conversation_summary_block,
            summary_refs=summary_refs,
            working_topic_block=working_topic_block,
            topic_refs=topic_refs,
            recent_verbatim_seed=(
                recent_verbatim_seed if self._curate_recent_verbatim_seed else []
            ),
        )
        items = result.items[: self._budget.max_curated_items]
        while items:
            block = InitialContextPackageCurator.render_block(items)
            if self._estimate(block) <= self._budget.curated_block_budget_tokens:
                return items, block, result.warnings
            items = items[:-1]
        warnings = list(result.warnings)
        if result.items:
            warnings.append("curation_trimmed_builder_budget")
        return [], "", warnings

    async def _build_current_state_block(
        self,
        *,
        user_id: str,
        workspace_id: str | None,
        conversation_id: str | None,
        user_persona_id: str | None,
        platform_id: str | None,
        character_id: str | None,
        incognito: bool,
        remember_across_chats: bool,
        remember_across_devices: bool,
        resolved_policy: ResolvedRetrievalPolicy,
        active_space_id: str | None,
        active_space_boundary_mode: str | None,
        active_mind_id: str | None,
        mind_topology: str | None,
        active_embodiment_id: str | None,
        active_realm_id: str | None,
    ) -> tuple[str, list[dict[str, Any]]]:
        rows = await self._fetch_visible_memory_rows(
            user_id=user_id,
            object_types=[MemoryObjectType.STATE_SNAPSHOT],
            statuses=[MemoryStatus.ACTIVE],
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
            remember_across_chats=remember_across_chats,
            remember_across_devices=remember_across_devices,
            resolved_policy=resolved_policy,
            active_space_id=active_space_id,
            active_space_boundary_mode=active_space_boundary_mode,
            active_mind_id=active_mind_id,
            mind_topology=mind_topology,
            active_embodiment_id=active_embodiment_id,
            active_realm_id=active_realm_id,
            limit=self._budget.max_state_rows,
        )
        refs_by_memory_id = await self._memory_source_refs_by_id(user_id, rows)
        lines = ["[Current State]"]
        refs: list[dict[str, Any]] = []
        for row in rows:
            memory_id = str(row["id"])
            payload = row.get("payload_json") or {}
            if not isinstance(payload, dict):
                payload = {}
            entries = payload.items() if payload else (("state", row.get("canonical_text")),)
            emitted_for_row = False
            for key, value in entries:
                line = self._state_line(str(key), value, row)
                if not line:
                    continue
                trial = "\n".join([*lines, line])
                if self._estimate(trial) > self._budget.current_state_block_budget_tokens:
                    break
                lines.append(line)
                emitted_for_row = True
            if emitted_for_row:
                refs.extend(refs_by_memory_id.get(memory_id) or [self._base_memory_ref(row)])
        if len(lines) == 1:
            return "", []
        return "\n".join(lines), refs

    def _state_line(self, key: str, value: Any, row: Mapping[str, Any]) -> str:
        if isinstance(value, (dict, list)):
            rendered_value = json_utils.dumps(value, sort_keys=True)
        else:
            rendered_value = str(value)
        text = self._compact_text(rendered_value, self._budget.max_state_value_chars)
        if not text:
            return ""
        memory_id = str(row.get("id") or "")
        return f"- {key}: {text} (source: {memory_id})"

    async def _fetch_visible_memory_rows(
        self,
        *,
        user_id: str,
        object_types: list[MemoryObjectType],
        statuses: list[MemoryStatus],
        workspace_id: str | None,
        conversation_id: str | None,
        user_persona_id: str | None,
        platform_id: str | None,
        character_id: str | None,
        incognito: bool,
        remember_across_chats: bool,
        remember_across_devices: bool,
        resolved_policy: ResolvedRetrievalPolicy,
        active_space_id: str | None,
        active_space_boundary_mode: str | None,
        active_mind_id: str | None,
        mind_topology: str | None,
        active_embodiment_id: str | None,
        active_realm_id: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        namespace_clause, namespace_parameters = self._memory_namespace_clause(
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
            remember_across_chats=remember_across_chats,
            remember_across_devices=remember_across_devices,
            resolved_policy=resolved_policy,
        )
        if not namespace_clause:
            return []
        space_clause, space_parameters = space_visibility_sql_clause_for_context(
            active_space_id=active_space_id,
            active_space_boundary_mode=active_space_boundary_mode,
            alias="mo",
        )
        mind_clause, mind_parameters = mind_visibility_sql_clause_for_context(
            active_mind_id=active_mind_id,
            mind_topology=mind_topology,
            alias="mo",
            allow_overseer_grants=True,
        )
        embodiment_clause, embodiment_parameters = embodiment_visibility_sql_clause_for_context(
            active_embodiment_id=active_embodiment_id,
            alias="mo",
        )
        realm_clause, realm_parameters = realm_visibility_sql_clause_for_context(
            active_realm_id=active_realm_id,
            alias="mo",
            allowed_bridge_modes=APPLICABLE_REALM_BRIDGE_MODES,
        )
        type_placeholders = ", ".join("?" for _ in object_types)
        status_placeholders = ", ".join("?" for _ in statuses)
        rows = await self._fetch_all(
            """
            SELECT mo.*
            FROM memory_objects AS mo
            WHERE mo.user_id = ?
              AND mo.object_type IN ({type_placeholders})
              AND mo.status IN ({status_placeholders})
              AND mo.archived_by_conversation_id IS NULL
              AND {visibility_clause}
              AND {namespace_clause}
              AND {space_clause}
              AND {mind_clause}
              AND {embodiment_clause}
              AND {realm_clause}
              AND mo.privacy_level <= ?
              AND {sensitivity_filter}
              AND {intimacy_filter}
            ORDER BY
              COALESCE(mo.vitality, 0.0) DESC,
              COALESCE(mo.tension_score, 0.0) DESC,
              COALESCE(mo.stability, 0.0) DESC,
              COALESCE(mo.confidence, 0.0) DESC,
              COALESCE(mo.maya_score, 0.0) ASC,
              mo.updated_at DESC,
              mo.id ASC
            LIMIT ?
            """.format(
                type_placeholders=type_placeholders,
                status_placeholders=status_placeholders,
                visibility_clause=conversation_visibility_clause("mo"),
                namespace_clause=namespace_clause,
                space_clause=space_clause,
                mind_clause=mind_clause,
                embodiment_clause=embodiment_clause,
                realm_clause=realm_clause,
                sensitivity_filter=MemoryObjectRepository.sensitivity_filter_clause(
                    gates_enabled=False,
                    allow_private_sensitivity=resolved_policy.allow_private_sensitivity,
                    table_alias="mo",
                ),
                intimacy_filter=memory_object_intimacy_sql_clause(
                    "mo",
                    allow_intimacy_context=resolved_policy.allow_intimacy_context,
                ),
            ),
            (
                user_id,
                *(object_type.value for object_type in object_types),
                *(status.value for status in statuses),
                conversation_id,
                *namespace_parameters,
                *space_parameters,
                *mind_parameters,
                *embodiment_parameters,
                *realm_parameters,
                resolved_policy.privacy_ceiling,
                limit,
            ),
        )
        await annotate_realm_bridge_modes_for_rows(
            self._connection,
            rows,
            active_realm_id=active_realm_id,
        )
        await annotate_overseer_grants_for_rows(
            self._connection,
            rows,
            active_mind_id=active_mind_id,
            mind_topology=mind_topology,
        )
        return rows

    def _merge_profile_rows(
        self,
        *row_groups: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for rows in row_groups:
            for row in rows:
                memory_id = str(row.get("id") or "")
                if not memory_id or memory_id in seen:
                    continue
                seen.add(memory_id)
                merged.append(row)
        return merged

    def _memory_namespace_clause(
        self,
        *,
        workspace_id: str | None,
        conversation_id: str | None,
        user_persona_id: str | None,
        platform_id: str | None,
        character_id: str | None,
        incognito: bool,
        remember_across_chats: bool,
        remember_across_devices: bool,
        resolved_policy: ResolvedRetrievalPolicy,
    ) -> tuple[str, list[Any]]:
        scopes = MemoryObjectRepository.canonical_retrieval_scopes(
            list(resolved_policy.allowed_scopes)
        )
        if conversation_id is None:
            scopes = [scope for scope in scopes if scope is not MemoryScope.CHAT]
        if platform_id is not None:
            clauses, parameters = MemoryObjectRepository.namespace_visibility_clauses(
                scopes,
                user_persona_id=user_persona_id,
                platform_id=platform_id,
                character_id=character_id,
                conversation_id=conversation_id or "",
                remember_across_chats=remember_across_chats,
                remember_across_devices=remember_across_devices,
                incognito=incognito,
                allow_private_sensitivity=resolved_policy.allow_private_sensitivity,
                table_alias="mo",
            )
            return (" AND ".join(clauses), parameters) if clauses else ("", [])

        scope_expr = (
            "CASE "
            "WHEN mo.scope_canonical IS NOT NULL THEN mo.scope_canonical "
            "WHEN mo.scope IN ('conversation', 'ephemeral_session') THEN 'chat' "
            "WHEN mo.scope = 'workspace' THEN 'character' "
            "WHEN mo.scope IN ('global_user', 'assistant_mode') THEN 'user' "
            "ELSE mo.scope END"
        )
        clauses: list[str] = []
        parameters: list[Any] = []
        for scope in scopes:
            if scope is MemoryScope.USER:
                clauses.append(
                    f"({scope_expr} = 'user' "
                    "AND mo.user_persona_id IS ? "
                    "AND mo.platform_locked = 0)"
                )
                parameters.append(user_persona_id)
            elif scope is MemoryScope.CHARACTER and (character_id is not None or workspace_id is not None):
                clauses.append(
                    f"({scope_expr} = 'character' "
                    "AND mo.user_persona_id IS ? "
                    "AND mo.platform_locked = 0 "
                    "AND ("
                    "(? IS NOT NULL AND mo.character_id = ?) "
                    "OR (? IS NOT NULL AND mo.character_id IS NULL AND mo.workspace_id = ?)"
                    "))"
                )
                parameters.extend(
                    [
                        user_persona_id,
                        character_id,
                        character_id,
                        workspace_id,
                        workspace_id,
                    ]
                )
            elif scope is MemoryScope.CHAT and conversation_id is not None:
                clauses.append(
                    f"({scope_expr} = 'chat' "
                    "AND mo.user_persona_id IS ? "
                    "AND mo.conversation_id = ?)"
                )
                parameters.extend([user_persona_id, conversation_id])
        return ("(" + " OR ".join(clauses) + ")", parameters) if clauses else ("", [])

    async def _memory_source_refs_by_id(
        self,
        user_id: str,
        rows: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        memory_ids = [str(row["id"]) for row in rows]
        refs = {memory_id: [self._base_memory_ref(row)] for memory_id, row in zip(memory_ids, rows)}
        if not memory_ids:
            return refs
        placeholders = ", ".join("?" for _ in memory_ids)
        evidence_rows = await self._fetch_all(
            f"""
            SELECT edge.id AS support_edge_id,
                   edge.memory_id,
                   edge.support_kind,
                   edge.evidence_polarity,
                   edge.speaker_relation_to_subject,
                   edge.confidence,
                   span.id AS evidence_span_id,
                   span.conversation_id,
                   span.message_id,
                   span.span_role,
                   span.seq,
                   span.occurred_at,
                   span.created_at
            FROM memory_support_edges AS edge
            LEFT JOIN memory_evidence_spans AS span
              ON span.support_edge_id = edge.id
             AND span.user_id = edge.user_id
             AND span.memory_id = edge.memory_id
            WHERE edge.user_id = ?
              AND edge.memory_id IN ({placeholders})
              AND edge.status = 'active'
            ORDER BY edge.memory_id ASC, edge.updated_at DESC, edge.id ASC, span.created_at ASC, span.id ASC
            """,
            (user_id, *memory_ids),
        )
        for row in evidence_rows:
            memory_id = str(row["memory_id"])
            refs.setdefault(memory_id, [])
            refs[memory_id].append(
                {
                    "source_kind": "memory_evidence",
                    "memory_id": memory_id,
                    "support_edge_id": row.get("support_edge_id"),
                    "support_kind": row.get("support_kind"),
                    "evidence_polarity": row.get("evidence_polarity"),
                    "speaker_relation_to_subject": row.get(
                        "speaker_relation_to_subject"
                    ),
                    "confidence": row.get("confidence"),
                    "evidence_span_id": row.get("evidence_span_id"),
                    "conversation_id": row.get("conversation_id"),
                    "message_id": row.get("message_id"),
                    "span_role": row.get("span_role"),
                    "seq": row.get("seq"),
                    "occurred_at": row.get("occurred_at"),
                }
            )
        return refs

    def _base_memory_ref(self, row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "source_kind": "memory_object",
            "memory_id": row.get("id"),
            "object_type": row.get("object_type"),
            "source_kind_row": row.get("source_kind"),
            "conversation_id": row.get("conversation_id"),
            "updated_at": row.get("updated_at"),
        }

    async def _build_summary_block(
        self,
        *,
        user_id: str,
        conversation_id: str | None,
        resolved_policy: ResolvedRetrievalPolicy,
    ) -> tuple[str, list[dict[str, Any]]]:
        if conversation_id is None:
            return "", []
        sensitivity_clause = (
            "sv.sensitivity IN ('public', 'private')"
            if resolved_policy.allow_private_sensitivity
            else "sv.sensitivity = 'public'"
        )
        intimacy_clause = (
            "1 = 1"
            if resolved_policy.allow_intimacy_context
            else "sv.intimacy_boundary = 'ordinary'"
        )
        rows = await self._fetch_all(
            """
            SELECT sv.*
            FROM summary_views AS sv
            WHERE sv.user_id = ?
              AND sv.conversation_id = ?
              AND sv.summary_kind = ?
              AND {sensitivity_clause}
              AND {intimacy_clause}
            ORDER BY sv.source_message_end_seq DESC, sv.id ASC
            LIMIT ?
            """.format(
                sensitivity_clause=sensitivity_clause,
                intimacy_clause=intimacy_clause,
            ),
            (
                user_id,
                conversation_id,
                SummaryViewKind.CONVERSATION_CHUNK.value,
                self._budget.max_summary_chunks,
            ),
        )
        rows = list(reversed(rows))
        lines: list[str] = []
        refs: list[dict[str, Any]] = []
        for row in rows:
            rendered = format_chunk_summary(row)
            if not rendered:
                continue
            trial = "\n\n".join([*lines, rendered])
            if self._estimate(trial) > self._budget.summary_block_budget_tokens:
                break
            lines.append(rendered)
            refs.append(
                {
                    "source_kind": "summary_view",
                    "summary_id": row.get("id"),
                    "conversation_id": row.get("conversation_id"),
                    "source_message_start_seq": row.get("source_message_start_seq"),
                    "source_message_end_seq": row.get("source_message_end_seq"),
                    "created_at": row.get("created_at"),
                }
            )
        return "\n\n".join(lines), refs

    async def _build_topic_block(
        self,
        *,
        user_id: str,
        conversation_id: str | None,
        resolved_policy: ResolvedRetrievalPolicy,
    ) -> tuple[str, list[dict[str, Any]]]:
        if conversation_id is None:
            return "", []
        from atagia.core.topic_repository import TopicRepository

        snapshot = await TopicRepository(self._connection, self._clock).get_topic_snapshot(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        filtered_snapshot = filter_topic_working_set_snapshot(
            snapshot,
            allow_intimacy_context=resolved_policy.allow_intimacy_context,
            privacy_ceiling=resolved_policy.privacy_ceiling,
        )
        if not filtered_snapshot:
            return "", []
        bounded_snapshot = self._bounded_topic_snapshot(
            filtered_snapshot,
            active_limit=len(list(filtered_snapshot.get("active_topics") or [])),
            parked_limit=min(2, len(list(filtered_snapshot.get("parked_topics") or []))),
        )
        block = render_topic_working_set_block(
            bounded_snapshot,
            allow_intimacy_context=resolved_policy.allow_intimacy_context,
            privacy_ceiling=resolved_policy.privacy_ceiling,
        )
        if self._estimate(block) > self._budget.topic_block_budget_tokens:
            bounded_snapshot = self._bounded_topic_snapshot(
                filtered_snapshot,
                active_limit=1,
                parked_limit=0,
                text_limit=96,
            )
            block = render_topic_working_set_block(
                bounded_snapshot,
                allow_intimacy_context=resolved_policy.allow_intimacy_context,
                privacy_ceiling=resolved_policy.privacy_ceiling,
            )
        if self._estimate(block) > self._budget.topic_block_budget_tokens:
            return "", []
        refs = [
            {
                "source_kind": "conversation_topic",
                "topic_id": topic.get("id"),
                "conversation_id": conversation_id,
                "source_refs": list(topic.get("source_refs") or []),
            }
            for topic in [
                *list(bounded_snapshot.get("active_topics") or []),
                *list(bounded_snapshot.get("parked_topics") or []),
            ]
        ]
        return block, refs

    def _bounded_topic_snapshot(
        self,
        snapshot: Mapping[str, Any],
        *,
        active_limit: int,
        parked_limit: int,
        text_limit: int = 180,
    ) -> dict[str, Any]:
        return {
            "active_topics": [
                self._bounded_topic(topic, text_limit=text_limit)
                for topic in list(snapshot.get("active_topics") or [])[:active_limit]
            ],
            "parked_topics": [
                self._bounded_topic(topic, text_limit=text_limit)
                for topic in list(snapshot.get("parked_topics") or [])[:parked_limit]
            ],
            "freshness": dict(snapshot.get("freshness") or {}),
        }

    def _bounded_topic(
        self,
        topic: Mapping[str, Any],
        *,
        text_limit: int,
    ) -> dict[str, Any]:
        bounded = dict(topic)
        for key in ("title", "summary", "active_goal"):
            value = bounded.get(key)
            if value is not None:
                bounded[key] = self._compact_text(str(value), text_limit)
        for key in ("decisions", "open_questions"):
            value = bounded.get(key)
            if isinstance(value, list):
                bounded[key] = [
                    self._compact_text(str(item), text_limit)
                    for item in value[:2]
                    if str(item).strip()
                ]
        return bounded

    async def _build_recent_verbatim_seed(
        self,
        *,
        user_id: str,
        conversation_id: str | None,
    ) -> list[dict[str, Any]]:
        if conversation_id is None:
            return []
        rows = await self._message_repository.get_recent_messages(
            conversation_id,
            user_id,
            limit=self._budget.max_recent_seed_messages,
        )
        seeds: list[dict[str, Any]] = []
        for row in rows:
            text = self._compact_text(
                message_text_for_context(row),
                self._budget.max_recent_seed_chars,
            )
            if not text:
                continue
            seeds.append(
                {
                    "source_kind": "message",
                    "message_id": row.get("id"),
                    "conversation_id": row.get("conversation_id"),
                    "seq": row.get("seq"),
                    "role": row.get("role"),
                    "text": text,
                    "occurred_at": row.get("occurred_at"),
                    "skip_by_default": bool(row.get("skip_by_default")),
                }
            )
        return seeds

    def _render_coordinate_context_block(self, markers: Mapping[str, Any]) -> str:
        lines = ["[Coordinate Context]"]
        conversation = markers.get("conversation")
        if isinstance(conversation, dict):
            lines.append(
                "- conversation: "
                + self._compact_text(
                    json_utils.dumps(
                        {
                            key: conversation.get(key)
                            for key in (
                                "id",
                                "status",
                                "incognito",
                                "temporary",
                                "purge_on_close",
                                "isolated_mode",
                            )
                        },
                        sort_keys=True,
                    ),
                    320,
                )
            )
        for label, key in (
            ("presence", "presence"),
            ("space", "space"),
            ("mind", "mind"),
            ("embodiment", "embodiment"),
        ):
            row = markers.get(key)
            if isinstance(row, dict) and row:
                lines.append(f"- {label}: {self._coordinate_label(row)}")
        ojocentauri = markers.get("ojocentauri")
        if isinstance(ojocentauri, dict) and ojocentauri.get("overseer_mind_id"):
            grant_count = len(list(ojocentauri.get("grants") or []))
            lines.append(
                f"- ojocentauri: overseer={ojocentauri.get('overseer_mind_id')} grants={grant_count}"
            )
        realm = markers.get("realm")
        if isinstance(realm, dict):
            row = realm.get("row")
            if isinstance(row, dict) and row:
                bridge_count = len(list(realm.get("bridges") or []))
                lines.append(f"- realm: {self._coordinate_label(row)} bridges={bridge_count}")
        missing = markers.get("missing")
        if missing:
            lines.append(f"- missing_coordinate_markers: {', '.join(map(str, missing))}")
        return "\n".join(lines) if len(lines) > 1 else ""

    def _coordinate_label(self, row: Mapping[str, Any]) -> str:
        display = self._optional_text(row.get("display_name"))
        identifier = self._optional_text(row.get("id"))
        mode = (
            self._optional_text(row.get("boundary_mode"))
            or self._optional_text(row.get("kind"))
            or self._optional_text(row.get("cross_embodiment_mode"))
            or self._optional_text(row.get("cross_realm_mode"))
        )
        parts = [display or identifier or "unknown"]
        if identifier and display and identifier != display:
            parts.append(f"id={identifier}")
        if mode:
            parts.append(f"mode={mode}")
        updated_at = self._optional_text(row.get("updated_at"))
        if updated_at:
            parts.append(f"updated={updated_at}")
        return "; ".join(parts)

    def _space_boundary_mode_from_signature(
        self,
        markers: Mapping[str, Any],
    ) -> str | None:
        space = markers.get("space")
        if not isinstance(space, Mapping):
            return None
        return self._optional_text(space.get("boundary_mode"))

    def _enforce_package_budget(
        self,
        blocks: InitialContextPackageBlocks,
    ) -> tuple[InitialContextPackageBlocks, int, int, list[str]]:
        dropped_profile = 0
        dropped_curated = 0
        warnings: list[str] = []
        current = blocks
        while (
            self._estimate_blocks_tokens(current) > self._budget.package_budget_tokens
            and current.profile_items
        ):
            profile_items = list(current.profile_items[:-1])
            dropped_profile += 1
            profile_block = self._profile_block_text(profile_items)
            current = current.model_copy(
                update={
                    "profile_items": profile_items,
                    "prepared_memory_profile_block": profile_block,
                    "source_counts": {
                        **current.source_counts,
                        "profile_items": len(profile_items),
                    },
                }
            )
        if dropped_profile:
            warnings.append("package_budget_trimmed_profile_items")

        while (
            self._estimate_blocks_tokens(current) > self._budget.package_budget_tokens
            and current.curated_items
        ):
            curated_items = list(current.curated_items[:-1])
            dropped_curated += 1
            curated_block = InitialContextPackageCurator.render_block(curated_items)
            current = current.model_copy(
                update={
                    "curated_items": curated_items,
                    "curated_orientation_block": curated_block,
                    "source_counts": {
                        **current.source_counts,
                        "curated_items": len(curated_items),
                    },
                    "empty_markers": {
                        **current.empty_markers,
                        "curated_orientation_empty": not curated_items,
                    },
                }
            )
        if dropped_curated:
            warnings.append("package_budget_trimmed_curated_items")

        while (
            self._estimate_blocks_tokens(current) > self._budget.package_budget_tokens
            and current.recent_verbatim_seed
        ):
            recent_seed = list(current.recent_verbatim_seed[1:])
            current = current.model_copy(
                update={
                    "recent_verbatim_seed": recent_seed,
                    "source_counts": {
                        **current.source_counts,
                        "recent_verbatim_seed": len(recent_seed),
                    },
                    "empty_markers": {
                        **current.empty_markers,
                        "recent_verbatim_seed_empty": not recent_seed,
                    },
                }
            )
            if "package_budget_trimmed_recent_seed" not in warnings:
                warnings.append("package_budget_trimmed_recent_seed")

        for block_name, count_key, marker_key, warning in (
            (
                "conversation_summary_block",
                "conversation_summaries",
                "conversation_summary_empty",
                "package_budget_trimmed_conversation_summary",
            ),
            (
                "working_topic_block",
                "working_topics",
                "working_topic_empty",
                "package_budget_trimmed_working_topic",
            ),
            (
                "current_state_block",
                "current_state_refs",
                "current_state_empty",
                "package_budget_trimmed_current_state",
            ),
        ):
            if self._estimate_blocks_tokens(current) <= self._budget.package_budget_tokens:
                break
            if not getattr(current, block_name):
                continue
            current = current.model_copy(
                update={
                    block_name: "",
                    "source_counts": {
                        **current.source_counts,
                        count_key: 0,
                    },
                    "empty_markers": {
                        **current.empty_markers,
                        marker_key: True,
                    },
                }
            )
            warnings.append(warning)

        if self._estimate_blocks_tokens(current) > self._budget.package_budget_tokens:
            warnings.append("package_budget_exceeded_after_trimming")
        return current, dropped_profile, dropped_curated, warnings

    def _estimate_blocks_tokens(self, blocks: InitialContextPackageBlocks) -> int:
        text_parts = [
            blocks.contract_block,
            blocks.curated_orientation_block,
            blocks.prepared_memory_profile_block,
            blocks.current_state_block,
            blocks.coordinate_context_block,
            blocks.conversation_summary_block,
            blocks.working_topic_block,
            json_utils.dumps(blocks.recent_verbatim_seed, sort_keys=True),
        ]
        return sum(self._estimate(part) for part in text_parts if part)

    def _estimate(self, text: str) -> int:
        return ContextComposer.estimate_tokens(text or "")

    def _empty_markers(
        self,
        *,
        profile_items: list[InitialContextPackageProfileItem],
        curated_items: list[InitialContextPackageProfileItem],
        current_state_block: str,
        conversation_summary_block: str,
        working_topic_block: str,
        recent_verbatim_seed: list[dict[str, Any]],
        conversation_id: str | None,
    ) -> dict[str, bool]:
        markers = {
            "prepared_memory_profile_empty": not profile_items,
            "curated_orientation_empty": not curated_items,
            "current_state_empty": not bool(current_state_block.strip()),
        }
        if conversation_id is None:
            markers.update(
                {
                    "same_chat_history_known_empty": True,
                    "conversation_summary_empty": True,
                    "working_topic_empty": True,
                    "recent_verbatim_seed_empty": True,
                }
            )
        else:
            markers.update(
                {
                    "same_chat_history_known_empty": not recent_verbatim_seed,
                    "conversation_summary_empty": not bool(
                        conversation_summary_block.strip()
                    ),
                    "working_topic_empty": not bool(working_topic_block.strip()),
                    "recent_verbatim_seed_empty": not recent_verbatim_seed,
                }
            )
        return markers

    def _source_counts(
        self,
        *,
        profile_items: list[InitialContextPackageProfileItem],
        curated_items: list[InitialContextPackageProfileItem],
        contract_refs: list[dict[str, Any]],
        state_refs: list[dict[str, Any]],
        summary_refs: list[dict[str, Any]],
        topic_refs: list[dict[str, Any]],
        recent_verbatim_seed: list[dict[str, Any]],
    ) -> dict[str, int]:
        return {
            "contract_dimensions": len(contract_refs),
            "curated_items": len(curated_items),
            "profile_items": len(profile_items),
            "current_state_refs": len(state_refs),
            "conversation_summaries": len(summary_refs),
            "working_topics": len(topic_refs),
            "recent_verbatim_seed": len(recent_verbatim_seed),
        }

    def _scope_metadata(self, row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "user_id": row.get("user_id"),
            "scope": row.get("scope"),
            "scope_canonical": row.get("scope_canonical"),
            "user_persona_id": row.get("user_persona_id"),
            "platform_id": row.get("platform_id"),
            "character_id": row.get("character_id"),
            "workspace_id": row.get("workspace_id"),
            "conversation_id": row.get("conversation_id"),
            "privacy_level": row.get("privacy_level"),
            "sensitivity": row.get("sensitivity"),
        }

    def _coordinate_visibility_metadata(self, row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "active_presence_id": row.get("active_presence_id"),
            "source_presence_id": row.get("source_presence_id"),
            "presence_cluster_id": row.get("presence_cluster_id"),
            "space_id": row.get("space_id"),
            "space_boundary_mode": row.get("space_boundary_mode"),
            "memory_owner_id": row.get("memory_owner_id"),
            "source_mind_id": row.get("source_mind_id"),
            "mind_relation": row.get("mind_relation"),
            "mind_grant_kind": row.get("mind_grant_kind"),
            "mind_grant_target_kind": row.get("mind_grant_target_kind"),
            "mind_grant_target_id": row.get("mind_grant_target_id"),
            "mind_grant_visibility": row.get("mind_grant_visibility"),
            "embodiment_id": row.get("embodiment_id"),
            "realm_id": row.get("realm_id"),
            "realm_relation": row.get("realm_relation"),
            "realm_bridge_mode": row.get("realm_bridge_mode"),
        }

    def _freshness_metadata(self, row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at"),
            "valid_from": row.get("valid_from"),
            "valid_to": row.get("valid_to"),
            "temporal_type": row.get("temporal_type"),
            "tension_score": row.get("tension_score"),
            "tension_updated_at": row.get("tension_updated_at"),
            "maya_score": row.get("maya_score"),
            "vitality": row.get("vitality"),
            "stability": row.get("stability"),
            "confidence": row.get("confidence"),
        }

    def _profile_reason_category(self, row: Mapping[str, Any]) -> str:
        payload = row.get("payload_json")
        if isinstance(payload, dict):
            for key in ("reason_category", "category", "facet", "dimension_name"):
                value = self._optional_text(payload.get(key))
                if value is not None:
                    return value[:64]
        object_type = self._optional_text(row.get("object_type"))
        return object_type or "memory_profile"

    def _profile_item_status(
        self,
        row: Mapping[str, Any],
    ) -> str:
        if row.get("valid_to") is not None:
            return "historical"
        status = self._optional_text(row.get("status"))
        if status == MemoryStatus.SUPERSEDED.value:
            return "superseded"
        if status == MemoryStatus.REVIEW_REQUIRED.value:
            return "ambiguous"
        payload = row.get("payload_json")
        if isinstance(payload, dict):
            hinted = self._optional_text(
                payload.get("initial_context_status")
                or payload.get("profile_status")
                or payload.get("status_hint")
            )
            if hinted in {"current", "historical", "superseded", "ambiguous"}:
                return hinted
        if self._bounded_float(row.get("tension_score")) >= (
            self._budget.ambiguous_tension_threshold
        ):
            return "ambiguous"
        return "current"

    def _salience(self, row: Mapping[str, Any]) -> float:
        vitality = self._bounded_float(row.get("vitality"))
        stability = self._bounded_float(row.get("stability"))
        confidence = self._bounded_float(row.get("confidence"))
        tension = self._bounded_float(row.get("tension_score"))
        maya_score = max(0.0, float(row.get("maya_score") or 0.0))
        maya_penalty = min(0.12, maya_score * 0.04)
        score = (
            (0.42 * vitality)
            + (0.24 * stability)
            + (0.24 * confidence)
            + (0.10 * tension)
            - maya_penalty
        )
        profile_status = self._profile_item_status(row)
        if profile_status == "ambiguous":
            score = max(score, 0.45 + (0.25 * tension))
        if profile_status in {"historical", "superseded"}:
            score = max(score, 0.35 + (0.20 * tension))
        return max(0.0, min(1.0, score))

    def _historical_row_is_relevant(self, row: Mapping[str, Any]) -> bool:
        if self._bounded_float(row.get("tension_score")) >= (
            self._budget.historical_tension_threshold
        ):
            return True
        payload = row.get("payload_json")
        if isinstance(payload, dict):
            for key in (
                "successor_memory_id",
                "superseded_by_memory_id",
                "replacement_memory_id",
                "resolved_by_memory_id",
            ):
                if self._optional_text(payload.get(key)) is not None:
                    return True
        reference_time = (
            self._parse_datetime(row.get("valid_to"))
            or self._parse_datetime(row.get("tension_updated_at"))
            or self._parse_datetime(row.get("updated_at"))
        )
        if reference_time is None:
            return False
        age_seconds = (self._clock.now() - reference_time).total_seconds()
        return 0 <= age_seconds <= (
            self._budget.max_historical_profile_age_days * 24 * 60 * 60
        )

    @staticmethod
    def _bounded_float(value: Any) -> float:
        try:
            parsed = float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, parsed))

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, str) and value.strip():
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        else:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    async def _fetch_user_preferences(self, user_id: str) -> dict[str, Any]:
        row = await self._fetch_one(
            """
            SELECT id,
                   remember_across_chats,
                   remember_across_devices,
                   memory_privacy_mode
            FROM users
            WHERE id = ?
            """,
            (user_id,),
        )
        if row is None:
            raise ValueError("user_id must exist before building an initial context package")
        return row

    async def _fetch_conversation(
        self,
        *,
        user_id: str,
        conversation_id: str,
    ) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM conversations
            WHERE user_id = ?
              AND id = ?
            """,
            (user_id, conversation_id),
        )

    @staticmethod
    def _resolve_user_bool(value: bool | None, stored: Any, *, default: bool) -> bool:
        if value is not None:
            return bool(value)
        if stored is None:
            return default
        return bool(stored)

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        return bool(value)

    @staticmethod
    def _optional_text(value: Any) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @staticmethod
    def _compact_text(value: str, max_chars: int) -> str:
        normalized = " ".join(value.split())
        if len(normalized) <= max_chars:
            return normalized
        return normalized[: max(0, max_chars - 3)].rstrip() + "..."

    @staticmethod
    def _scope_rank(scope: str) -> int:
        if scope in {MemoryScope.CHAT.value, MemoryScope.CONVERSATION.value, MemoryScope.EPHEMERAL_SESSION.value}:
            return 3
        if scope in {MemoryScope.CHARACTER.value, MemoryScope.WORKSPACE.value}:
            return 2
        return 1
