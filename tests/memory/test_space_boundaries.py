"""Tests for Space boundary retrieval behavior."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import MemoryObjectRepository, UserRepository
from atagia.core.space_repository import SpaceRepository
from atagia.core.verbatim_pin_repository import VerbatimPinRepository
from atagia.memory.candidate_search import CandidateSearch
from atagia.models.schemas_memory import (
    MemoryObjectType,
    MemoryScope,
    MemorySensitivity,
    MemorySourceKind,
    MemoryStatus,
    PlannedSubQuery,
    RetrievalPlan,
    SpaceBoundaryMode,
    VerbatimPinTargetKind,
)

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


def _plan(
    *,
    active_space_id: str | None,
    active_space_boundary_mode: SpaceBoundaryMode | None,
) -> RetrievalPlan:
    return RetrievalPlan(
        original_query="alpha",
        assistant_mode_id="coding_debug",
        conversation_id="cnv_1",
        platform_id="default",
        active_space_id=active_space_id,
        active_space_boundary_mode=active_space_boundary_mode,
        fts_queries=["alpha"],
        sub_query_plans=[PlannedSubQuery(text="alpha", fts_queries=["alpha"])],
        scope_filter=[MemoryScope.USER],
        status_filter=[MemoryStatus.ACTIVE],
        max_candidates=20,
        max_context_items=20,
        privacy_ceiling=3,
        retrieval_levels=[0],
    )


@pytest.mark.asyncio
async def test_candidate_search_applies_space_boundary_modes() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 5, 11, tzinfo=timezone.utc))
    try:
        await UserRepository(connection, clock).create_user("usr_1")
        spaces = SpaceRepository(connection, clock)
        memories = MemoryObjectRepository(connection, clock)
        for space_id, mode in (
            ("space_focus", SpaceBoundaryMode.FOCUS),
            ("space_severance", SpaceBoundaryMode.SEVERANCE),
            ("space_vault", SpaceBoundaryMode.PRIVACY_VAULT),
            ("space_tagged", SpaceBoundaryMode.TAGGED),
        ):
            await spaces.resolve_space(
                owner_user_id="usr_1",
                space_id=space_id,
                boundary_mode=mode,
                display_name=space_id,
                source_kind="explicit",
                source_id=space_id,
            )
            await memories.create_memory_object(
                user_id="usr_1",
                object_type=MemoryObjectType.EVIDENCE,
                scope=MemoryScope.USER,
                canonical_text=f"alpha memory in {space_id}",
                source_kind=MemorySourceKind.EXTRACTED,
                confidence=0.9,
                privacy_level=0,
                sensitivity=MemorySensitivity.PUBLIC,
                memory_id=f"mem_{space_id}",
                space_id=space_id,
                space_boundary_mode=mode.value,
            )
        await memories.create_memory_object(
            user_id="usr_1",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.USER,
            canonical_text="alpha memory outside any space",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.9,
            privacy_level=0,
            sensitivity=MemorySensitivity.PUBLIC,
            memory_id="mem_global",
        )

        search = CandidateSearch(connection, clock)

        outside_ids = {
            row["id"]
            for row in await search.search(
                _plan(active_space_id=None, active_space_boundary_mode=None),
                "usr_1",
            )
        }
        assert outside_ids == {"mem_global", "mem_space_focus", "mem_space_tagged"}

        focus_ids = {
            row["id"]
            for row in await search.search(
                _plan(
                    active_space_id="space_focus",
                    active_space_boundary_mode=SpaceBoundaryMode.FOCUS,
                ),
                "usr_1",
            )
        }
        assert focus_ids == {"mem_global", "mem_space_focus", "mem_space_tagged"}

        severance_ids = {
            row["id"]
            for row in await search.search(
                _plan(
                    active_space_id="space_severance",
                    active_space_boundary_mode=SpaceBoundaryMode.SEVERANCE,
                ),
                "usr_1",
            )
        }
        assert severance_ids == {"mem_space_severance"}

        vault_ids = {
            row["id"]
            for row in await search.search(
                _plan(
                    active_space_id="space_vault",
                    active_space_boundary_mode=SpaceBoundaryMode.PRIVACY_VAULT,
                ),
                "usr_1",
            )
        }
        assert vault_ids == {"mem_global", "mem_space_tagged", "mem_space_vault"}

        tagged_ids = {
            row["id"]
            for row in await search.search(
                _plan(
                    active_space_id="space_tagged",
                    active_space_boundary_mode=SpaceBoundaryMode.TAGGED,
                ),
                "usr_1",
            )
        }
        assert tagged_ids == {"mem_global", "mem_space_focus", "mem_space_tagged"}
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_candidate_search_applies_space_boundaries_to_verbatim_pins() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 5, 11, tzinfo=timezone.utc))
    try:
        await UserRepository(connection, clock).create_user("usr_1")
        spaces = SpaceRepository(connection, clock)
        pins = VerbatimPinRepository(connection, clock)
        for space_id, mode in (
            ("space_vault", SpaceBoundaryMode.PRIVACY_VAULT),
            ("space_severance", SpaceBoundaryMode.SEVERANCE),
        ):
            await spaces.resolve_space(
                owner_user_id="usr_1",
                space_id=space_id,
                boundary_mode=mode,
                display_name=space_id,
                source_kind="explicit",
                source_id=space_id,
            )
            await pins.create_verbatim_pin(
                user_id="usr_1",
                scope=MemoryScope.USER,
                target_kind=VerbatimPinTargetKind.TEXT_SPAN,
                target_id=f"target_{space_id}",
                pin_id=f"pin_{space_id}",
                canonical_text=f"alpha exact pin in {space_id}",
                index_text="alpha exact pin",
                privacy_level=0,
                created_by="usr_1",
                platform_id="default",
                scope_canonical=MemoryScope.USER.value,
                space_id=space_id,
                space_boundary_mode=mode,
            )

        search = CandidateSearch(connection, clock)
        verbatim_plan = _plan(
            active_space_id=None,
            active_space_boundary_mode=None,
        ).model_copy(
            update={
                "raw_context_access_mode": "verbatim",
                "scope_filter": [MemoryScope.USER],
            }
        )

        outside_ids = {
            row["id"]
            for row in await search.search(verbatim_plan, "usr_1")
        }
        assert outside_ids == set()

        vault_candidates = await search.search(
            verbatim_plan.model_copy(
                update={
                    "active_space_id": "space_vault",
                    "active_space_boundary_mode": SpaceBoundaryMode.PRIVACY_VAULT,
                }
            ),
            "usr_1",
        )
        assert {row["id"] for row in vault_candidates} == {"pin_space_vault"}
        assert vault_candidates[0]["space_id"] == "space_vault"
        assert vault_candidates[0]["space_boundary_mode"] == SpaceBoundaryMode.PRIVACY_VAULT.value

        severance_candidates = await search.search(
            verbatim_plan.model_copy(
                update={
                    "active_space_id": "space_severance",
                    "active_space_boundary_mode": SpaceBoundaryMode.SEVERANCE,
                }
            ),
            "usr_1",
        )
        assert {row["id"] for row in severance_candidates} == {"pin_space_severance"}
        assert severance_candidates[0]["space_id"] == "space_severance"
        assert severance_candidates[0]["space_boundary_mode"] == SpaceBoundaryMode.SEVERANCE.value
    finally:
        await connection.close()
