"""Validation helpers for the graph/relation retrieval stress set."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from benchmarks.atagia_bench.adapter import AtagiaBenchAdapter, AtagiaBenchDataset


GRAPH_STRESS_DATA_DIR = Path(__file__).resolve().parent / "data"

REQUIRED_GRAPH_STRESS_TAGS = frozenset(
    {
        "graph_stress_multi_hop",
        "graph_stress_temporal_order",
        "graph_stress_connection_lookup",
        "graph_stress_supersession",
        "graph_stress_broad_list",
        "graph_stress_artifact_task",
    }
)


class GraphStressDatasetError(ValueError):
    """Raised when the graph stress dataset is structurally invalid."""


def load_graph_stress_dataset(data_dir: str | Path | None = None) -> AtagiaBenchDataset:
    """Load the graph stress set using the Atagia-bench data contract."""
    return AtagiaBenchAdapter(data_dir or GRAPH_STRESS_DATA_DIR).load()


def validate_graph_stress_dataset(dataset: AtagiaBenchDataset) -> dict[str, Any]:
    """Validate graph stress coverage and return a compact summary."""
    question_ids: set[str] = set()
    tag_counts: Counter[str] = Counter()
    conversation_count = 0

    for persona_data in dataset.personas:
        conversation_count += len(persona_data.conversations)
        turn_ids = {
            turn.turn_id
            for conversation in persona_data.conversations
            for turn in conversation.turns
        }
        for question in persona_data.questions:
            if question.question_id in question_ids:
                raise GraphStressDatasetError(
                    f"Duplicate graph stress question id: {question.question_id}"
                )
            question_ids.add(question.question_id)
            question_tags = set(question.category_tags)
            if "graph_stress" not in question_tags:
                raise GraphStressDatasetError(
                    f"Graph stress question missing graph_stress tag: {question.question_id}"
                )
            if not question_tags.intersection(REQUIRED_GRAPH_STRESS_TAGS):
                raise GraphStressDatasetError(
                    "Graph stress question lacks a required stress-family tag: "
                    f"{question.question_id}"
                )
            missing_turn_ids = [
                turn_id
                for turn_id in question.evidence_turn_ids
                if turn_id not in turn_ids
            ]
            if missing_turn_ids:
                raise GraphStressDatasetError(
                    f"Question {question.question_id} references missing evidence turns: "
                    f"{', '.join(sorted(missing_turn_ids))}"
                )
            tag_counts.update(question.category_tags)

    missing_tags = sorted(REQUIRED_GRAPH_STRESS_TAGS.difference(tag_counts))
    if missing_tags:
        raise GraphStressDatasetError(
            f"Graph stress dataset is missing required tags: {', '.join(missing_tags)}"
        )

    return {
        "dataset_name": dataset.name,
        "persona_count": len(dataset.personas),
        "conversation_count": conversation_count,
        "question_count": len(question_ids),
        "tag_counts": dict(sorted(tag_counts.items())),
        "required_graph_stress_tags": sorted(REQUIRED_GRAPH_STRESS_TAGS),
    }
