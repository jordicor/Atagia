"""Central temperature policy for Atagia LLM calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


MIN_COMPLETION_TEMPERATURE = 0.1


@dataclass(frozen=True, slots=True)
class PurposeTemperature:
    """Default completion temperature for a stable LLM request purpose."""

    value: float
    reason: str


_VERIFIER_TEMPERATURE = PurposeTemperature(
    0.2,
    "mechanical verifier/classifier",
)
_SEMANTIC_TEMPERATURE = PurposeTemperature(
    0.5,
    "semantic understanding",
)
_SYNTHESIS_TEMPERATURE = PurposeTemperature(
    0.7,
    "planning or synthesis",
)
_CHAT_TEMPERATURE = PurposeTemperature(
    1.0,
    "chat answer generation",
)


PURPOSE_TEMPERATURES: dict[str, PurposeTemperature] = {
    "answer_abstention_legitimacy_verification": _VERIFIER_TEMPERATURE,
    "answer_evidence_use_verification": _VERIFIER_TEMPERATURE,
    "answer_postcondition_verification": _VERIFIER_TEMPERATURE,
    "applicability_date_card": _VERIFIER_TEMPERATURE,
    "applicability_relevance_card": PurposeTemperature(0.4, "nuanced candidate scoring card"),
    "applicability_scoring": PurposeTemperature(0.4, "nuanced candidate scoring"),
    "benchmark_broad_list_coverage_judge": _VERIFIER_TEMPERATURE,
    "benchmark_exact_product_prompt_replay": _CHAT_TEMPERATURE,
    "benchmark_fixed_context_answer_generation": _CHAT_TEMPERATURE,
    "benchmark_grader_abstention": _VERIFIER_TEMPERATURE,
    "benchmark_grader_gated_fact": _VERIFIER_TEMPERATURE,
    "benchmark_grader_privacy_off_fact": _VERIFIER_TEMPERATURE,
    "benchmark_grader_supersession": _VERIFIER_TEMPERATURE,
    "benchmark_judge": _VERIFIER_TEMPERATURE,
    "belief_revision": _SEMANTIC_TEMPERATURE,
    "chat_reply": _CHAT_TEMPERATURE,
    "compaction_summary_judge": _VERIFIER_TEMPERATURE,
    "compaction_summary_refine": _SEMANTIC_TEMPERATURE,
    "consequence_detection": _VERIFIER_TEMPERATURE,
    "consequence_action_card": _VERIFIER_TEMPERATURE,
    "consequence_gate_card": _VERIFIER_TEMPERATURE,
    "consequence_language_card": _VERIFIER_TEMPERATURE,
    "consequence_link_card": _VERIFIER_TEMPERATURE,
    "consequence_outcome_card": _VERIFIER_TEMPERATURE,
    "consequence_sentiment_card": _VERIFIER_TEMPERATURE,
    "consequence_tendency_inference": _SEMANTIC_TEMPERATURE,
    "consent_confirmation_intent": _VERIFIER_TEMPERATURE,
    "content_language_backfill": _VERIFIER_TEMPERATURE,
    "context_cache_signal_detection": _VERIFIER_TEMPERATURE,
    "contract_projection": _SEMANTIC_TEMPERATURE,
    "coverage_expansion": _SYNTHESIS_TEMPERATURE,
    "episode_synthesis": _SYNTHESIS_TEMPERATURE,
    "evaluation_contract_compliance": _VERIFIER_TEMPERATURE,
    "export_anonymization_rewrite": _SEMANTIC_TEMPERATURE,
    "export_anonymization_verify": _VERIFIER_TEMPERATURE,
    "extraction_watchdog": _VERIFIER_TEMPERATURE,
    "graph_projection": _SEMANTIC_TEMPERATURE,
    "initial_context_package_curation": PurposeTemperature(
        0.4,
        "grounded profile curation",
    ),
    "intent_classifier_claim_key_equivalence": _VERIFIER_TEMPERATURE,
    "intent_classifier_explicit": _VERIFIER_TEMPERATURE,
    "locomo_source_context_replay_answer": _CHAT_TEMPERATURE,
    "locomo_source_context_answer_probe": _CHAT_TEMPERATURE,
    "memory_extraction": _SEMANTIC_TEMPERATURE,
    "memory_extraction_candidate_card": _SEMANTIC_TEMPERATURE,
    "memory_extraction_kind_scope_card": _SEMANTIC_TEMPERATURE,
    "memory_extraction_evidence_card": _SEMANTIC_TEMPERATURE,
    "memory_extraction_index_card": _SEMANTIC_TEMPERATURE,
    "memory_extraction_temporal_card": _SEMANTIC_TEMPERATURE,
    "memory_extraction_belief_card": _SEMANTIC_TEMPERATURE,
    "memory_extraction_coverage_members_card": _SEMANTIC_TEMPERATURE,
    "metrics_computer_contract_compliance": _VERIFIER_TEMPERATURE,
    "model_casting_applicability": PurposeTemperature(0.4, "nuanced candidate scoring"),
    "model_casting_compactor_segmentation": PurposeTemperature(
        0.4,
        "semantic segmentation",
    ),
    "model_casting_intent": _VERIFIER_TEMPERATURE,
    "model_casting_staleness": _VERIFIER_TEMPERATURE,
    "model_casting_text_chunker": PurposeTemperature(0.4, "semantic segmentation"),
    "need_detection_needs_card": _VERIFIER_TEMPERATURE,
    "need_detection_language_card": _VERIFIER_TEMPERATURE,
    "need_detection_memory_card": _VERIFIER_TEMPERATURE,
    "need_detection_exact_card": _VERIFIER_TEMPERATURE,
    "need_detection_shape_card": _VERIFIER_TEMPERATURE,
    "need_detection_facets_card": _VERIFIER_TEMPERATURE,
    "need_detection_callback_card": _VERIFIER_TEMPERATURE,
    "need_detection_search_words_card": _SEMANTIC_TEMPERATURE,
    "need_detection_search_words_other_language_card": _SEMANTIC_TEMPERATURE,
    "retrieval_surface_generation_dry_run": _SYNTHESIS_TEMPERATURE,
    "summary_chunk_segmentation_ranges_card": PurposeTemperature(
        0.4,
        "semantic segmentation card",
    ),
    "summary_chunk_segmentation_summaries_card": PurposeTemperature(
        0.4,
        "grounded chunk summary card",
    ),
    "summary_privacy_gate_judge": _VERIFIER_TEMPERATURE,
    "summary_privacy_gate_refine": PurposeTemperature(0.4, "privacy-safe refinement"),
    "text_chunking_level1": PurposeTemperature(0.4, "semantic segmentation"),
    "thematic_profile_synthesis": _SYNTHESIS_TEMPERATURE,
    "third_party_bench_answer_generation": _CHAT_TEMPERATURE,
    "topic_working_set_update": _SYNTHESIS_TEMPERATURE,
    "topic_working_set_route_card": _VERIFIER_TEMPERATURE,
    "topic_working_set_content_card": _SYNTHESIS_TEMPERATURE,
    "topic_working_set_boundary_card": _VERIFIER_TEMPERATURE,
    "user_language_profile_observed_card": _VERIFIER_TEMPERATURE,
    "user_language_profile_preference_card": _VERIFIER_TEMPERATURE,
    "user_language_profile_ability_card": _VERIFIER_TEMPERATURE,
    "user_language_profile_norm_card": _VERIFIER_TEMPERATURE,
    "workspace_rollup_synthesis": _SYNTHESIS_TEMPERATURE,
}


def purpose_temperature(purpose: Any) -> PurposeTemperature | None:
    """Return the configured default temperature for a request purpose."""
    if not isinstance(purpose, str):
        return None
    return PURPOSE_TEMPERATURES.get(purpose.strip())
