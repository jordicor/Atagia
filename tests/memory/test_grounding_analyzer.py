"""Tests for grounding analysis."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import MemoryObjectRepository, UserRepository
from atagia.memory.grounding_analyzer import GroundingAnalyzer
from atagia.models.schemas_memory import ComposedContext, MemoryObjectType, MemoryScope, MemorySourceKind

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


async def _build_runtime():
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 4, 5, 13, 0, tzinfo=timezone.utc))
    users = UserRepository(connection, clock)
    memories = MemoryObjectRepository(connection, clock)
    await users.create_user("usr_1")
    return connection, memories


async def _seed_memory(
    memories: MemoryObjectRepository,
    *,
    memory_id: str,
    source_kind: MemorySourceKind,
    maya_score: float,
) -> None:
    await memories.create_memory_object(
        user_id="usr_1",
        object_type=MemoryObjectType.EVIDENCE,
        scope=MemoryScope.GLOBAL_USER,
        canonical_text=memory_id,
        source_kind=source_kind,
        confidence=0.8,
        privacy_level=0,
        maya_score=maya_score,
        memory_id=memory_id,
    )


@pytest.mark.asyncio
async def test_grounding_analyzer_classifies_items_and_computes_ratios() -> None:
    connection, memories = await _build_runtime()
    try:
        await _seed_memory(memories, memory_id="mem_verbatim", source_kind=MemorySourceKind.VERBATIM, maya_score=0.4)
        await _seed_memory(memories, memory_id="mem_extracted", source_kind=MemorySourceKind.EXTRACTED, maya_score=0.8)
        await _seed_memory(memories, memory_id="mem_inferred", source_kind=MemorySourceKind.INFERRED, maya_score=1.2)
        await _seed_memory(memories, memory_id="mem_summary", source_kind=MemorySourceKind.SUMMARIZED, maya_score=2.0)

        report = await GroundingAnalyzer(connection).analyze(
            ComposedContext(
                contract_block="",
                workspace_block="",
                memory_block="",
                state_block="",
                selected_memory_ids=["mem_verbatim", "mem_extracted", "mem_inferred", "mem_summary"],
                total_tokens_estimate=10,
                budget_tokens=100,
                items_included=4,
                items_dropped=0,
            ),
            "usr_1",
        )

        by_id = {item.memory_id: item for item in report.items}
        assert by_id["mem_verbatim"].grounding_level.value == "grounded"
        assert by_id["mem_extracted"].grounding_level.value == "derived"
        assert by_id["mem_inferred"].grounding_level.value == "inferred"
        assert by_id["mem_summary"].grounding_level.value == "summary"
        assert report.grounded_ratio == pytest.approx(0.25)
        assert report.high_maya_items == ["mem_inferred", "mem_summary"]
    finally:
        await connection.close()
