"""Declared environment-variable sources for every ``Settings`` field.

``Settings.from_env`` reads each field from one or more named environment
variables. Provenance reporting needs that mapping by NAME: a field whose env
var is set is env-sourced even when the value happens to equal the code
default, and comparing values cannot tell those apart.

This map is the declaration of that wiring. It is kept in lockstep with
``Settings.from_env`` by ``tests/core/test_settings_env_vars.py``, which
re-derives the mapping from the ``from_env`` AST and fails the build on any
drift, so a new env var cannot be read without being declared here.
"""

from __future__ import annotations

from atagia.services.model_resolution import COMPONENT_SPECS

# Per-component families are enumerated from the component registry rather than
# spelled out, so adding a component cannot leave its env vars undeclared.
COMPONENT_MODEL_ENV_VARS: tuple[str, ...] = tuple(
    spec.env_var for spec in COMPONENT_SPECS
)
COMPONENT_EXAMPLES_ENV_VARS: tuple[str, ...] = tuple(
    spec.examples_env_var for spec in COMPONENT_SPECS
)
INTIMACY_COMPONENT_MODEL_ENV_VARS: tuple[str, ...] = tuple(
    spec.intimacy_env_var for spec in COMPONENT_SPECS
)


SETTINGS_ENV_VARS: dict[str, tuple[str, ...]] = {
    "sqlite_path": ("ATAGIA_SQLITE_PATH",),
    "migrations_path": ("ATAGIA_MIGRATIONS_PATH",),
    "manifests_path": ("ATAGIA_MANIFESTS_PATH",),
    "operational_profiles_path": ("ATAGIA_OPERATIONAL_PROFILES_PATH",),
    "artifact_blob_storage_kind": ("ATAGIA_ARTIFACT_BLOB_STORAGE_KIND",),
    "artifact_blob_storage_path": ("ATAGIA_ARTIFACT_BLOB_STORAGE_PATH",),
    "storage_backend": ("ATAGIA_STORAGE_BACKEND",),
    "redis_url": ("ATAGIA_REDIS_URL",),
    "anthropic_api_key": ("ATAGIA_ANTHROPIC_API_KEY",),
    "openai_api_key": ("ATAGIA_OPENAI_API_KEY",),
    "openrouter_api_key": ("ATAGIA_OPENROUTER_API_KEY",),
    "kimi_api_key": ("ATAGIA_KIMI_API_KEY",),
    "minimax_api_key": ("ATAGIA_MINIMAX_API_KEY",),
    "typesafe_api_key": ("ATAGIA_TYPESAFE_API_KEY",),
    "anthropic_base_url": ("ATAGIA_ANTHROPIC_BASE_URL",),
    "anthropic_request_timeout_seconds": ("ATAGIA_ANTHROPIC_REQUEST_TIMEOUT_SECONDS",),
    "llm_request_timeout_seconds": ("ATAGIA_LLM_REQUEST_TIMEOUT_SECONDS",),
    "openai_base_url": ("ATAGIA_OPENAI_BASE_URL",),
    "openai_embedding_base_url": ("ATAGIA_OPENAI_EMBEDDING_BASE_URL",),
    "kimi_base_url": ("ATAGIA_KIMI_BASE_URL",),
    "minimax_base_url": ("ATAGIA_MINIMAX_BASE_URL",),
    "openrouter_base_url": ("ATAGIA_OPENROUTER_BASE_URL",),
    "inference_access_mode": ("ATAGIA_INFERENCE_ACCESS_MODE",),
    "local_llm_endpoints_file": ("ATAGIA_LOCAL_LLM_ENDPOINTS_FILE",),
    "zero_cost_openrouter_profile": ("ATAGIA_ZERO_COST_OPENROUTER_PROFILE",),
    "openrouter_site_url": ("ATAGIA_OPENROUTER_SITE_URL",),
    "openrouter_app_name": ("ATAGIA_OPENROUTER_APP_NAME",),
    "llm_chat_model": ("ATAGIA_LLM_CHAT_MODEL",),
    "llm_max_concurrent_requests_per_provider": (
        "ATAGIA_LLM_MAX_CONCURRENT_REQUESTS_PER_PROVIDER",
    ),
    "llm_forced_global_model": ("ATAGIA_LLM_FORCED_GLOBAL_MODEL",),
    "llm_ingest_model": ("ATAGIA_LLM_INGEST_MODEL",),
    "llm_retrieval_model": ("ATAGIA_LLM_RETRIEVAL_MODEL",),
    "llm_finite_decisions_enabled": ("ATAGIA_LLM_FINITE_DECISIONS_ENABLED",),
    "llm_finite_decision_model": ("ATAGIA_LLM_FINITE_DECISION_MODEL",),
    "llm_component_models": COMPONENT_MODEL_ENV_VARS,
    "card_examples_enabled": ("ATAGIA_CARD_EXAMPLES_ENABLED",),
    "llm_component_examples": COMPONENT_EXAMPLES_ENV_VARS,
    "llm_intimacy_ingest_model": ("ATAGIA_LLM_INTIMACY_INGEST_MODEL",),
    "llm_intimacy_retrieval_model": ("ATAGIA_LLM_INTIMACY_RETRIEVAL_MODEL",),
    "llm_intimacy_component_models": INTIMACY_COMPONENT_MODEL_ENV_VARS,
    "llm_intimacy_proactive_routing_enabled": (
        "ATAGIA_LLM_INTIMACY_PROACTIVE_ROUTING_ENABLED",
    ),
    "llm_structured_output_retry_attempts": (
        "ATAGIA_LLM_STRUCTURED_OUTPUT_RETRY_ATTEMPTS",
    ),
    "llm_structured_output_rescue_enabled": (
        "ATAGIA_LLM_STRUCTURED_OUTPUT_RESCUE_ENABLED",
    ),
    "llm_structured_output_rescue_model": (
        "ATAGIA_LLM_STRUCTURED_OUTPUT_RESCUE_MODEL",
    ),
    "llm_technical_recovery_enabled": ("ATAGIA_LLM_TECHNICAL_RECOVERY_ENABLED",),
    "llm_output_limit_retry_attempts": ("ATAGIA_LLM_OUTPUT_LIMIT_RETRY_ATTEMPTS",),
    "llm_runaway_watchdog_enabled": ("ATAGIA_LLM_RUNAWAY_WATCHDOG_ENABLED",),
    "llm_runaway_min_elapsed_seconds": ("ATAGIA_LLM_RUNAWAY_MIN_ELAPSED_SECONDS",),
    "llm_runaway_min_output_tokens": ("ATAGIA_LLM_RUNAWAY_MIN_OUTPUT_TOKENS",),
    "llm_runaway_check_interval_tokens": ("ATAGIA_LLM_RUNAWAY_CHECK_INTERVAL_TOKENS",),
    "llm_runaway_max_checks": ("ATAGIA_LLM_RUNAWAY_MAX_CHECKS",),
    "llm_runaway_hard_abort_min_output_tokens": (
        "ATAGIA_LLM_RUNAWAY_HARD_ABORT_MIN_OUTPUT_TOKENS",
    ),
    "llm_runaway_min_repeat_count": ("ATAGIA_LLM_RUNAWAY_MIN_REPEAT_COUNT",),
    "llm_runaway_min_repeat_ratio_tokens": (
        "ATAGIA_LLM_RUNAWAY_MIN_REPEAT_RATIO_TOKENS",
    ),
    "llm_runaway_output_input_ratio": ("ATAGIA_LLM_RUNAWAY_OUTPUT_INPUT_RATIO",),
    "llm_runaway_hard_output_input_ratio": (
        "ATAGIA_LLM_RUNAWAY_HARD_OUTPUT_INPUT_RATIO",
    ),
    "llm_debug_io_enabled": ("ATAGIA_DEBUG_LLM_IO",),
    "llm_debug_io_dir": ("ATAGIA_DEBUG_LLM_IO_DIR",),
    "llm_debug_io_purposes": ("ATAGIA_DEBUG_LLM_IO_PURPOSES",),
    "llm_debug_io_raw": ("ATAGIA_DEBUG_LLM_IO_RAW",),
    "llm_debug_io_max_chars": ("ATAGIA_DEBUG_LLM_IO_MAX_CHARS",),
    "diagnostic_capture_enabled": ("ATAGIA_DIAGNOSTIC_CAPTURE_ENABLED",),
    "diagnostic_capture_dir": ("ATAGIA_DIAGNOSTIC_CAPTURE_DIR",),
    "diagnostic_capture_max_blob_bytes": ("ATAGIA_DIAGNOSTIC_CAPTURE_MAX_BLOB_BYTES",),
    "diagnostic_capture_max_session_bytes": ("ATAGIA_DIAGNOSTIC_CAPTURE_MAX_SESSION_BYTES",),
    "service_mode": ("ATAGIA_SERVICE_MODE",),
    "service_api_key": ("ATAGIA_SERVICE_API_KEY",),
    "admin_api_key": ("ATAGIA_ADMIN_API_KEY",),
    "allow_admin_export_anonymization": ("ATAGIA_ALLOW_ADMIN_EXPORT_ANONYMIZATION",),
    "workers_enabled": ("ATAGIA_WORKERS_ENABLED",),
    "debug": ("ATAGIA_DEBUG",),
    "worker_circuit_breaker_enabled": ("ATAGIA_WORKER_CIRCUIT_BREAKER_ENABLED",),
    "worker_circuit_breaker_failure_threshold": (
        "ATAGIA_WORKER_CIRCUIT_BREAKER_FAILURE_THRESHOLD",
    ),
    "worker_circuit_breaker_window_seconds": (
        "ATAGIA_WORKER_CIRCUIT_BREAKER_WINDOW_SECONDS",
    ),
    "worker_circuit_breaker_min_failure_ratio": (
        "ATAGIA_WORKER_CIRCUIT_BREAKER_MIN_FAILURE_RATIO",
    ),
    "worker_transient_defer_seconds": ("ATAGIA_WORKER_TRANSIENT_DEFER_SECONDS",),
    "worker_transient_defer_max_seconds": (
        "ATAGIA_WORKER_TRANSIENT_DEFER_MAX_SECONDS",
    ),
    "worker_transient_defer_max_count": ("ATAGIA_WORKER_TRANSIENT_DEFER_MAX_COUNT",),
    "worker_transient_defer_max_age_seconds": (
        "ATAGIA_WORKER_TRANSIENT_DEFER_MAX_AGE_SECONDS",
    ),
    "worker_retry_backoff_initial_seconds": (
        "ATAGIA_WORKER_RETRY_BACKOFF_INITIAL_SECONDS",
    ),
    "worker_retry_backoff_max_seconds": ("ATAGIA_WORKER_RETRY_BACKOFF_MAX_SECONDS",),
    "worker_dispatch_visibility_seconds": (
        "ATAGIA_WORKER_DISPATCH_VISIBILITY_SECONDS",
    ),
    "worker_dispatch_sweep_interval_seconds": (
        "ATAGIA_WORKER_DISPATCH_SWEEP_INTERVAL_SECONDS",
    ),
    "worker_dispatch_batch_size": ("ATAGIA_WORKER_DISPATCH_BATCH_SIZE",),
    "worker_execution_lease_seconds": ("ATAGIA_WORKER_EXECUTION_LEASE_SECONDS",),
    "worker_execution_heartbeat_seconds": (
        "ATAGIA_WORKER_EXECUTION_HEARTBEAT_SECONDS",
    ),
    "worker_stream_reclaim_idle_seconds": (
        "ATAGIA_WORKER_STREAM_RECLAIM_IDLE_SECONDS",
    ),
    "service_process_count": ("ATAGIA_SERVICE_PROCESS_COUNT", "WEB_CONCURRENCY"),
    "llm_run_guard_enabled": ("ATAGIA_LLM_RUN_GUARD_ENABLED",),
    "llm_run_guard_mode": ("ATAGIA_LLM_RUN_GUARD_MODE",),
    "llm_run_guard_max_total_calls": ("ATAGIA_LLM_RUN_GUARD_MAX_TOTAL_CALLS",),
    "llm_run_guard_max_total_failed_calls": (
        "ATAGIA_LLM_RUN_GUARD_MAX_TOTAL_FAILED_CALLS",
    ),
    "llm_run_guard_max_failed_call_ratio": (
        "ATAGIA_LLM_RUN_GUARD_MAX_FAILED_CALL_RATIO",
    ),
    "llm_run_guard_failed_ratio_min_calls": (
        "ATAGIA_LLM_RUN_GUARD_FAILED_RATIO_MIN_CALLS",
    ),
    "llm_run_guard_max_failed_calls_per_purpose": (
        "ATAGIA_LLM_RUN_GUARD_MAX_FAILED_CALLS_PER_PURPOSE",
    ),
    "llm_run_guard_max_failed_ratio_per_purpose": (
        "ATAGIA_LLM_RUN_GUARD_MAX_FAILED_RATIO_PER_PURPOSE",
    ),
    "llm_run_guard_purpose_failure_ratio_min_calls": (
        "ATAGIA_LLM_RUN_GUARD_PURPOSE_FAILURE_RATIO_MIN_CALLS",
    ),
    "llm_run_guard_max_consecutive_failures_per_purpose": (
        "ATAGIA_LLM_RUN_GUARD_MAX_CONSECUTIVE_FAILURES_PER_PURPOSE",
    ),
    "llm_run_guard_max_total_tokens": ("ATAGIA_LLM_RUN_GUARD_MAX_TOTAL_TOKENS",),
    "llm_run_guard_max_reported_cost_usd": (
        "ATAGIA_LLM_RUN_GUARD_MAX_REPORTED_COST_USD",
    ),
    "llm_run_guard_health_window_calls": (
        "ATAGIA_LLM_RUN_GUARD_HEALTH_WINDOW_CALLS",
    ),
    "llm_run_guard_recovery_seconds": ("ATAGIA_LLM_RUN_GUARD_RECOVERY_SECONDS",),
    "bulk_ingest_llm_run_guard_enabled": ("ATAGIA_BULK_INGEST_LLM_RUN_GUARD_ENABLED",),
    "bulk_ingest_llm_run_guard_max_total_calls": (
        "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_MAX_TOTAL_CALLS",
    ),
    "bulk_ingest_llm_run_guard_max_total_failed_calls": (
        "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_MAX_TOTAL_FAILED_CALLS",
    ),
    "bulk_ingest_llm_run_guard_max_failed_call_ratio": (
        "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_MAX_FAILED_CALL_RATIO",
    ),
    "bulk_ingest_llm_run_guard_failed_ratio_min_calls": (
        "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_FAILED_RATIO_MIN_CALLS",
    ),
    "bulk_ingest_llm_run_guard_max_failed_calls_per_purpose": (
        "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_MAX_FAILED_CALLS_PER_PURPOSE",
    ),
    "bulk_ingest_llm_run_guard_max_failed_ratio_per_purpose": (
        "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_MAX_FAILED_RATIO_PER_PURPOSE",
    ),
    "bulk_ingest_llm_run_guard_purpose_failure_ratio_min_calls": (
        "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_PURPOSE_FAILURE_RATIO_MIN_CALLS",
    ),
    "bulk_ingest_llm_run_guard_max_consecutive_failures_per_purpose": (
        "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_MAX_CONSECUTIVE_FAILURES_PER_PURPOSE",
    ),
    "bulk_ingest_llm_run_guard_max_total_tokens": (
        "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_MAX_TOTAL_TOKENS",
    ),
    "bulk_ingest_llm_run_guard_max_reported_cost_usd": (
        "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_MAX_REPORTED_COST_USD",
    ),
    "bulk_ingest_llm_run_guard_max_wall_time_seconds": (
        "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_MAX_WALL_TIME_SECONDS",
    ),
    "google_api_key": (
        "ATAGIA_GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "GEMINI_KEY",
        "GOOGLE_API_KEY",
    ),
    "allow_insecure_http": ("ATAGIA_ALLOW_INSECURE_HTTP",),
    "default_language_code": ("ATAGIA_DEFAULT_LANGUAGE_CODE",),
    "consequence_detector_card_concurrency": (
        "ATAGIA_CONSEQUENCE_DETECTOR_CARD_CONCURRENCY",
    ),
    "compactor_summary_card_concurrency": (
        "ATAGIA_COMPACTOR_SUMMARY_CARD_CONCURRENCY",
    ),
    "applicability_scorer_card_concurrency": (
        "ATAGIA_APPLICABILITY_SCORER_CARD_CONCURRENCY",
    ),
    "embedding_backend": ("ATAGIA_EMBEDDING_BACKEND",),
    "embedding_model": ("ATAGIA_EMBEDDING_MODEL",),
    "embedding_dimension": ("ATAGIA_EMBEDDING_DIMENSION",),
    "embedding_vector_limit_cap": ("ATAGIA_EMBEDDING_VECTOR_LIMIT_CAP",),
    "embedding_search_overfetch_multiplier": (
        "ATAGIA_EMBEDDING_SEARCH_OVERFETCH_MULTIPLIER",
    ),
    "rrf_k": ("ATAGIA_RRF_K",),
    "recall_recovery_scoring_max_candidates": (
        "ATAGIA_RECALL_RECOVERY_SCORING_MAX_CANDIDATES",
    ),
    "broad_list_comparable_rrf_enabled": (
        "ATAGIA_BROAD_LIST_COMPARABLE_RRF_ENABLED",
    ),
    "fused_candidate_guard_ordering_enabled": (
        "ATAGIA_FUSED_CANDIDATE_GUARD_ORDERING_ENABLED",
    ),
    "memory_fts_canonical_bm25_weight": ("ATAGIA_MEMORY_FTS_CANONICAL_BM25_WEIGHT",),
    "memory_fts_index_bm25_weight": ("ATAGIA_MEMORY_FTS_INDEX_BM25_WEIGHT",),
    "lifecycle_decay_days": ("ATAGIA_LIFECYCLE_DECAY_DAYS",),
    "lifecycle_decay_rate": ("ATAGIA_LIFECYCLE_DECAY_RATE",),
    "lifecycle_archive_vitality": ("ATAGIA_LIFECYCLE_ARCHIVE_VITALITY",),
    "lifecycle_archive_confidence": ("ATAGIA_LIFECYCLE_ARCHIVE_CONFIDENCE",),
    "ephemeral_scoring_hours": ("ATAGIA_EPHEMERAL_SCORING_HOURS",),
    "lifecycle_ephemeral_ttl_hours": ("ATAGIA_LIFECYCLE_EPHEMERAL_TTL_HOURS",),
    "lifecycle_review_ttl_days": ("ATAGIA_LIFECYCLE_REVIEW_TTL_DAYS",),
    "promotion_conv_to_ws_min_conversations": (
        "ATAGIA_PROMOTION_CONV_TO_WS_MIN_CONVERSATIONS",
    ),
    "promotion_ws_to_global_min_sessions": (
        "ATAGIA_PROMOTION_WS_TO_GLOBAL_MIN_SESSIONS",
    ),
    "promotion_require_mode_consistency": (
        "ATAGIA_PROMOTION_REQUIRE_MODE_CONSISTENCY",
    ),
    "belief_tension_increment": ("ATAGIA_BELIEF_TENSION_INCREMENT",),
    "belief_tension_decrement": ("ATAGIA_BELIEF_TENSION_DECREMENT",),
    "belief_tension_threshold": ("ATAGIA_BELIEF_TENSION_THRESHOLD",),
    "skip_belief_revision": ("ATAGIA_SKIP_BELIEF_REVISION",),
    "skip_compaction": ("ATAGIA_SKIP_COMPACTION",),
    "episode_synthesis_max_episodes": ("ATAGIA_EPISODE_SYNTHESIS_MAX_EPISODES",),
    "opf_privacy_filter_enabled": ("ATAGIA_OPF_PRIVACY_FILTER_ENABLED",),
    "opf_primary_url": ("ATAGIA_OPF_PRIMARY_URL",),
    "opf_fallback_url": ("ATAGIA_OPF_FALLBACK_URL",),
    "opf_timeout_seconds": ("ATAGIA_OPF_TIMEOUT_SECONDS",),
    "privacy_validation_gate_enabled": ("ATAGIA_PRIVACY_VALIDATION_GATE_ENABLED",),
    "privacy_validation_gate_timeout_seconds": (
        "ATAGIA_PRIVACY_VALIDATION_GATE_TIMEOUT_SECONDS",
    ),
    "privacy_validation_gate_max_source_chars": (
        "ATAGIA_PRIVACY_VALIDATION_GATE_MAX_SOURCE_CHARS",
    ),
    "privacy_validation_gate_max_summaries_gated_per_job": (
        "ATAGIA_PRIVACY_VALIDATION_GATE_MAX_SUMMARIES_GATED_PER_JOB",
    ),
    "operational_high_risk_enabled": ("ATAGIA_OPERATIONAL_HIGH_RISK_ENABLED",),
    "operational_allowed_profiles": ("ATAGIA_OPERATIONAL_ALLOWED_PROFILES",),
    "context_cache_enabled": ("ATAGIA_CONTEXT_CACHE_ENABLED",),
    "initial_context_package_read_enabled": (
        "ATAGIA_INITIAL_CONTEXT_PACKAGE_READ_ENABLED",
    ),
    "initial_context_package_refresh_enabled": (
        "ATAGIA_INITIAL_CONTEXT_PACKAGE_REFRESH_ENABLED",
    ),
    "initial_context_package_curation_enabled": (
        "ATAGIA_INITIAL_CONTEXT_PACKAGE_CURATION_ENABLED",
    ),
    "initial_context_package_prompt_max_tokens": (
        "ATAGIA_INITIAL_CONTEXT_PACKAGE_PROMPT_MAX_TOKENS",
    ),
    "initial_context_package_profile_max_tokens": (
        "ATAGIA_INITIAL_CONTEXT_PACKAGE_PROFILE_MAX_TOKENS",
    ),
    "initial_context_package_total_max_tokens": (
        "ATAGIA_INITIAL_CONTEXT_PACKAGE_TOTAL_MAX_TOKENS",
    ),
    "initial_context_package_curated_block_max_tokens": (
        "ATAGIA_INITIAL_CONTEXT_PACKAGE_CURATED_BLOCK_MAX_TOKENS",
    ),
    "initial_context_package_curated_max_items": (
        "ATAGIA_INITIAL_CONTEXT_PACKAGE_CURATED_MAX_ITEMS",
    ),
    "initial_context_package_curation_max_output_tokens": (
        "ATAGIA_INITIAL_CONTEXT_PACKAGE_CURATION_MAX_OUTPUT_TOKENS",
    ),
    "context_cache_min_ttl_seconds": ("ATAGIA_CONTEXT_CACHE_MIN_TTL_SECONDS",),
    "context_cache_max_ttl_seconds": ("ATAGIA_CONTEXT_CACHE_MAX_TTL_SECONDS",),
    "temporary_default_ttl_seconds": ("ATAGIA_TEMPORARY_DEFAULT_TTL_SECONDS",),
    "temporary_default_purge_on_close": ("ATAGIA_TEMPORARY_DEFAULT_PURGE_ON_CLOSE",),
    "tombstone_retention_days": ("ATAGIA_TOMBSTONE_RETENTION_DAYS",),
    "erasure_purge_streams": ("ATAGIA_ERASURE_PURGE_STREAMS",),
    "disable_chunking_extraction": ("ATAGIA_DISABLE_CHUNKING_EXTRACTION",),
    "chunking_extraction_threshold_tokens": (
        "ATAGIA_CHUNKING_EXTRACTION_THRESHOLD_TOKENS",
    ),
    "extraction_watchdog_enabled": ("ATAGIA_EXTRACTION_WATCHDOG_ENABLED",),
    "extraction_watchdog_allow_different_provider": (
        "ATAGIA_EXTRACTION_WATCHDOG_ALLOW_DIFFERENT_PROVIDER",
    ),
    "extraction_watchdog_bounded_retry_max_items": (
        "ATAGIA_EXTRACTION_WATCHDOG_BOUNDED_RETRY_MAX_ITEMS",
    ),
    "extraction_watchdog_bounded_retry_max_output_tokens": (
        "ATAGIA_EXTRACTION_WATCHDOG_BOUNDED_RETRY_MAX_OUTPUT_TOKENS",
    ),
    "lifecycle_lazy_enabled": ("ATAGIA_LIFECYCLE_LAZY_ENABLED",),
    "lifecycle_min_interval_seconds": ("ATAGIA_LIFECYCLE_MIN_INTERVAL_SECONDS",),
    "lifecycle_busy_timeout_ms": ("ATAGIA_LIFECYCLE_BUSY_TIMEOUT_MS",),
    "lifecycle_busy_backoff_seconds": ("ATAGIA_LIFECYCLE_BUSY_BACKOFF_SECONDS",),
    "lifecycle_failure_backoff_seconds": ("ATAGIA_LIFECYCLE_FAILURE_BACKOFF_SECONDS",),
    "lifecycle_worker_enabled": ("ATAGIA_LIFECYCLE_WORKER_ENABLED",),
    "lifecycle_worker_interval_seconds": ("ATAGIA_LIFECYCLE_WORKER_INTERVAL_SECONDS",),
    "retrieval_packets_dry_run_enabled": ("ATAGIA_RETRIEVAL_PACKETS_DRY_RUN_ENABLED",),
    "retrieval_packets_write_enabled": ("ATAGIA_RETRIEVAL_PACKETS_WRITE_ENABLED",),
    "fact_facet_surfaces_enabled": ("ATAGIA_FACT_FACET_SURFACES_ENABLED",),
    "fact_facet_retrieval_enabled": ("ATAGIA_FACT_FACET_RETRIEVAL_ENABLED",),
    "fact_facet_structured_only": ("ATAGIA_FACT_FACET_STRUCTURED_ONLY",),
    "fact_facet_span_coadmission_enabled": (
        "ATAGIA_FACT_FACET_SPAN_COADMISSION_ENABLED",
    ),
    "fact_facet_retrieval_limit": ("ATAGIA_FACT_FACET_RETRIEVAL_LIMIT",),
    "fact_facet_retrieval_rrf_weight": ("ATAGIA_FACT_FACET_RETRIEVAL_RRF_WEIGHT",),
    "applicability_gate_mode": ("ATAGIA_APPLICABILITY_GATE_MODE",),
    "small_corpus_token_threshold_ratio": (
        "ATAGIA_SMALL_CORPUS_TOKEN_THRESHOLD_RATIO",
    ),
    "assistant_guidance_enabled": ("ATAGIA_ASSISTANT_GUIDANCE_ENABLED",),
    "response_mode": ("ATAGIA_RESPONSE_MODE",),
    "adaptive_retrieval": ("ATAGIA_ADAPTIVE_RETRIEVAL",),
    "answer_stance": ("ATAGIA_ANSWER_STANCE",),
    "answer_stance_prompt_variant": ("ATAGIA_ANSWER_STANCE_PROMPT_VARIANT",),
    "answer_postcondition_guard_enabled": (
        "ATAGIA_ANSWER_POSTCONDITION_GUARD_ENABLED",
    ),
    "answer_postcondition_retry_max_output_tokens": (
        "ATAGIA_ANSWER_POSTCONDITION_RETRY_MAX_OUTPUT_TOKENS",
    ),
    "context_envelope_budget_tokens": ("ATAGIA_CONTEXT_ENVELOPE_BUDGET_TOKENS",),
    "context_envelope_ratios": ("ATAGIA_CONTEXT_ENVELOPE_RATIOS",),
    "benchmark_disable_raw_recent_transcript": (
        "ATAGIA_BENCHMARK_DISABLE_RAW_RECENT_TRANSCRIPT",
    ),
    "recent_transcript_overage_ratio": ("ATAGIA_RECENT_TRANSCRIPT_OVERAGE_RATIO",),
    "topic_working_set_enabled": ("ATAGIA_TOPIC_WORKING_SET_ENABLED",),
    "topic_working_set_update_mode": ("ATAGIA_TOPIC_WORKING_SET_UPDATE_MODE",),
    "topic_working_set_refresh_message_lag": (
        "ATAGIA_TOPIC_WORKING_SET_REFRESH_MESSAGE_LAG",
    ),
    "topic_working_set_stale_message_lag": (
        "ATAGIA_TOPIC_WORKING_SET_STALE_MESSAGE_LAG",
    ),
    "topic_working_set_refresh_token_lag": (
        "ATAGIA_TOPIC_WORKING_SET_REFRESH_TOKEN_LAG",
    ),
    "topic_working_set_stale_token_lag": ("ATAGIA_TOPIC_WORKING_SET_STALE_TOKEN_LAG",),
    "topic_working_set_refresh_batch_messages": (
        "ATAGIA_TOPIC_WORKING_SET_REFRESH_BATCH_MESSAGES",
    ),
    "graph_projection_enabled": ("ATAGIA_GRAPH_PROJECTION_ENABLED",),
    "verbatim_evidence_search_enabled": ("ATAGIA_VERBATIM_EVIDENCE_SEARCH_ENABLED",),
    "verbatim_evidence_search_rrf_weight": (
        "ATAGIA_VERBATIM_EVIDENCE_SEARCH_RRF_WEIGHT",
    ),
    "verbatim_evidence_search_limit": ("ATAGIA_VERBATIM_EVIDENCE_SEARCH_LIMIT",),
    "verbatim_evidence_window_size": ("ATAGIA_VERBATIM_EVIDENCE_WINDOW_SIZE",),
    "verbatim_evidence_window_overlap": ("ATAGIA_VERBATIM_EVIDENCE_WINDOW_OVERLAP",),
    "openai_proxy_model_id": ("ATAGIA_PROXY_MODEL_ID",),
    "openai_proxy_upstream_model": ("ATAGIA_PROXY_UPSTREAM_MODEL",),
    "openai_proxy_default_mode": ("ATAGIA_PROXY_DEFAULT_MODE",),
    "openai_proxy_max_output_tokens": ("ATAGIA_PROXY_MAX_OUTPUT_TOKENS",),
    "request_max_body_bytes": ("ATAGIA_REQUEST_MAX_BODY_BYTES",),
    "request_max_message_text_bytes": ("ATAGIA_REQUEST_MAX_MESSAGE_TEXT_BYTES",),
    "request_max_attachments": ("ATAGIA_REQUEST_MAX_ATTACHMENTS",),
    "request_max_attachment_decoded_bytes": (
        "ATAGIA_REQUEST_MAX_ATTACHMENT_DECODED_BYTES",
    ),
    "request_max_attachments_decoded_bytes": (
        "ATAGIA_REQUEST_MAX_ATTACHMENTS_DECODED_BYTES",
    ),
    "request_max_metadata_bytes": ("ATAGIA_REQUEST_MAX_METADATA_BYTES",),
    "cors_allowed_origins": ("ATAGIA_CORS_ALLOWED_ORIGINS",),
}
