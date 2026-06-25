"""Benchmark activation flag manifest helpers."""

from __future__ import annotations

from typing import Any

from atagia.core.config import Settings


def benchmark_activation_flags(
    *,
    embedding_backend: str,
    answer_postcondition_guard_enabled: bool,
) -> dict[str, Any]:
    """Return the consolidated feature-activation block for run manifests."""
    settings = Settings.from_env()
    return {
        "fact_facet_surfaces_enabled": settings.fact_facet_surfaces_enabled,
        "fact_facet_retrieval_enabled": settings.fact_facet_retrieval_enabled,
        "applicability_gate_mode": settings.applicability_gate_mode,
        "answer_postcondition_guard_enabled": bool(
            answer_postcondition_guard_enabled
        ),
        "embedding_backend": str(embedding_backend or settings.embedding_backend),
        "graph_projection_enabled": settings.graph_projection_enabled,
        "response_mode": settings.response_mode,
        "adaptive_retrieval": settings.adaptive_retrieval,
    }
