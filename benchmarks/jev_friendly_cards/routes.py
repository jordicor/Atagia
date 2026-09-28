"""Explicit model routes for isolated A/B/C benchmark processes."""

from __future__ import annotations

from pathlib import Path

import atagia

from atagia.core.config import Settings
from atagia.services.model_resolution import (
    COMPONENTS_BY_ID,
    FINITE_DECISION_COMPONENT_IDS,
    resolve_component_model,
)

from benchmarks.jev_friendly_cards.protocol import ARMS


LUNA6 = "openrouter/openai/gpt-6-luna"
LUNA56 = "openrouter/openai/gpt-5.6-luna"
GEMINI = "openrouter/google/gemini-3.1-flash-lite"
JEV = "typesafe/jev-1.13.0"

_C_JEV_COMPONENTS = frozenset(
    {
        "extraction_kind",
        "extraction_scope",
        "extraction_confidence",
        "extraction_evidence",
        "extraction_evidence_support",
        "extraction_preserve_verbatim",
        "extraction_temporal_type",
        "extraction_member_identity",
        "intent_classifier",
        "need_detector_language",
        "need_detector_query_language",
        "need_detector_answer_language",
        "topic_title_decision",
        "topic_summary_decision",
        "topic_goal_decision",
        "topic_questions_decision",
        "topic_decisions_decision",
    }
)


def production_root() -> Path:
    return Path(atagia.__file__).resolve().parents[2]


def settings_for_arm(arm: str) -> Settings:
    if arm not in ARMS:
        raise ValueError(f"Unknown evaluation arm: {arm}")
    root = production_root()
    common_models = {
        "extractor": LUNA6,
        "extraction_evidence": LUNA6,
        "topic_working_set": LUNA56,
        "need_detector_language": GEMINI,
        "intent_classifier": LUNA56,
        "text_chunker": LUNA56,
        "belief_reviser": LUNA56,
        "extraction_watchdog": LUNA56,
    }
    shared = {
        "sqlite_path": ":memory:",
        "migrations_path": str(root / "src/atagia/resources/migrations"),
        "manifests_path": str(root / "src/atagia/resources/manifests"),
        "storage_backend": "inprocess",
        "redis_url": "redis://localhost:6379/0",
        "openai_api_key": None,
        "openrouter_api_key": None,
        "openrouter_site_url": "http://localhost",
        "openrouter_app_name": "Atagia decision comparison",
        "llm_chat_model": None,
        "service_mode": False,
        "service_api_key": None,
        "admin_api_key": None,
        "workers_enabled": False,
        "debug": False,
        "llm_max_concurrent_requests_per_provider": 2,
        "llm_structured_output_retry_attempts": 0,
        "llm_structured_output_rescue_enabled": False,
        "opf_privacy_filter_enabled": False,
        "retrieval_packets_dry_run_enabled": False,
        "retrieval_packets_write_enabled": False,
        "fact_facet_surfaces_enabled": False,
        "graph_projection_enabled": False,
    }
    if arm == "A_baseline_llm":
        return Settings(
            **shared,
            llm_component_models=common_models,
            llm_finite_decisions_enabled=False,
        )
    shared["topic_working_set_update_mode"] = "selective"
    if arm == "B_shared_llm":
        return Settings(
            **shared,
            llm_component_models=common_models,
            llm_finite_decisions_enabled=False,
        )

    reference = Settings(
        **shared,
        llm_component_models=common_models,
        llm_finite_decisions_enabled=False,
    )
    models = {
        **common_models,
        **{
            component: resolve_component_model(reference, component)
            for component in FINITE_DECISION_COMPONENT_IDS
        },
        **{component: JEV for component in _C_JEV_COMPONENTS},
    }
    return Settings(
        **shared,
        llm_component_models=models,
        llm_finite_decisions_enabled=True,
    )


def audited_routes(arm: str) -> dict[str, str]:
    settings = settings_for_arm(arm)
    return {
        component: resolve_component_model(settings, component)
        for component in COMPONENTS_BY_ID
    }
