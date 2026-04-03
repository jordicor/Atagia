"""Grounding analysis for composed retrieval context."""

from __future__ import annotations

from typing import Any

import aiosqlite

from atagia.core.repositories import _decode_json_columns
from atagia.models.schemas_memory import ComposedContext, MemorySourceKind
from atagia.models.schemas_replay import GroundingItem, GroundingLevel, GroundingReport


class GroundingAnalyzer:
    """Classify selected memories by grounding level."""

    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._connection = connection

    async def analyze(
        self,
        composed_context: ComposedContext | dict[str, Any],
        user_id: str,
    ) -> GroundingReport:
        if isinstance(composed_context, ComposedContext):
            selected_memory_ids = [str(memory_id) for memory_id in composed_context.selected_memory_ids]
        else:
            selected_memory_ids = [
                str(memory_id)
                for memory_id in (composed_context.get("selected_memory_ids") or [])
            ]
        if not selected_memory_ids:
            return GroundingReport(items=[], grounded_ratio=0.0, avg_maya_score=0.0, high_maya_items=[])

        placeholders = ", ".join("?" for _ in selected_memory_ids)
        cursor = await self._connection.execute(
            """
            SELECT *
            FROM memory_objects
            WHERE user_id = ?
              AND id IN ({placeholders})
            """.format(placeholders=placeholders),
            (user_id, *selected_memory_ids),
        )
        rows = await cursor.fetchall()
        by_id = {
            str(decoded["id"]): decoded
            for row in rows
            if (decoded := _decode_json_columns(row)) is not None
        }

        items: list[GroundingItem] = []
        for memory_id in selected_memory_ids:
            row = by_id.get(memory_id)
            if row is None:
                continue
            grounding_level = self._grounding_level(
                source_kind=str(row["source_kind"]),
                maya_score=float(row.get("maya_score", 0.0)),
            )
            items.append(
                GroundingItem(
                    memory_id=memory_id,
                    canonical_text=str(row.get("canonical_text", "")),
                    object_type=str(row.get("object_type", "")),
                    source_kind=str(row.get("source_kind", "")),
                    maya_score=float(row.get("maya_score", 0.0)),
                    grounding_level=grounding_level,
                )
            )

        if not items:
            return GroundingReport(items=[], grounded_ratio=0.0, avg_maya_score=0.0, high_maya_items=[])

        grounded_count = sum(1 for item in items if item.grounding_level is GroundingLevel.GROUNDED)
        avg_maya_score = sum(item.maya_score for item in items) / len(items)
        high_maya_items = [item.memory_id for item in items if item.maya_score > 1.0]
        return GroundingReport(
            items=items,
            grounded_ratio=grounded_count / len(items),
            avg_maya_score=avg_maya_score,
            high_maya_items=high_maya_items,
        )

    @staticmethod
    def _grounding_level(*, source_kind: str, maya_score: float) -> GroundingLevel:
        if source_kind == MemorySourceKind.VERBATIM.value and maya_score <= 0.5:
            return GroundingLevel.GROUNDED
        if source_kind == MemorySourceKind.EXTRACTED.value and maya_score <= 1.0:
            return GroundingLevel.DERIVED
        if source_kind == MemorySourceKind.INFERRED.value and maya_score <= 1.5:
            return GroundingLevel.INFERRED
        return GroundingLevel.SUMMARY
