"""Configuration loading from environment variables."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

from atagia.core.env import env_bool_optional
from atagia.core.language_codes import normalize_optional_iso_639_1_code
from atagia.memory.context_envelope import (
    CONTEXT_ENVELOPE_DEFAULT_RATIOS,
    allocate_context_envelope_budget,
)
from atagia.services.model_resolution import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_STRUCTURED_OUTPUT_RESCUE_MODEL,
    component_env_examples_from_env,
    component_env_models_from_env,
    intimacy_component_env_models_from_env,
    validate_finite_decision_configuration,
)


_DOTENV_LOADED = False
DEFAULT_LANGUAGE_CODE = "en"
RESPONSE_MODES = ("normal", "fast", "smart_fast")
ANSWER_STANCES = ("reactive", "proactive")
ANSWER_STANCE_PROMPT_VARIANTS = (
    "baseline",
    "template_v1",
    "template_v2",
    "template_v3",
    "category_hint",
    "exact_template",
    "no_substitute",
    "binary_gate",
    "if_else_gate",
    "label_precision",
    "strict_adjacent_silence",
    "liability_guard",
    "hard_exact_stop",
    "hard_no_extra",
    "partial_evidence_override",
    "stance_gate_combo",
)


def _repo_root() -> Path | None:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return None


def default_resource_path(name: str) -> str:
    """Return the single canonical packaged resource directory."""
    return str(Path(__file__).resolve().parents[1] / "resources" / name)


def configured_resource_path(name: str, configured: str | None) -> str:
    """Resolve configured resource directories safely across host cwd values."""
    if not configured:
        return default_resource_path(name)

    path = Path(configured).expanduser()
    if path.is_absolute():
        return str(path)

    if (Path.cwd() / path).exists():
        return str(path)

    repo_root = _repo_root()
    if repo_root is not None:
        repo_relative = repo_root / path
        if repo_relative.exists():
            return str(repo_relative)

    return str(path)


def _load_dotenv_once() -> None:
    """Load .env from the project root, idempotent across calls.

    Walks up from this file to find the project root (where pyproject.toml
    lives). Existing environment variables take precedence over .env values
    so explicit overrides at run time still win.
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True

    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        candidate = parent / "pyproject.toml"
        if candidate.exists():
            env_path = parent / ".env"
            if env_path.exists():
                load_dotenv(env_path, override=False)
            return


class _EnvReader:
    """Typed reads bound to one explicit environment mapping.

    ``Settings.from_env`` reads every value through this reader, so the whole
    configuration is a pure function of the mapping it is handed. Nothing here
    touches ``os.environ`` directly: a caller that wants the code-default
    baseline passes an empty mapping instead of mutating the process
    environment.
    """

    __slots__ = ("mapping",)

    def __init__(self, mapping: Mapping[str, str]) -> None:
        self.mapping = mapping

    def get(self, name: str, default: str | None = None) -> str | None:
        return self.mapping.get(name, default)

    def bool(self, name: str, default: bool) -> bool:
        value = env_bool_optional(name, self.mapping)
        return default if value is None else value

    def csv_tuple(self, name: str, default: tuple[str, ...]) -> tuple[str, ...]:
        value = self.mapping.get(name)
        if value is None:
            return default
        normalized = tuple(part.strip() for part in value.split(",") if part.strip())
        return normalized or default

    def optional_int(self, name: str) -> int | None:
        value = self.mapping.get(name)
        if value is None or not value.strip():
            return None
        return int(value)

    def optional_int_default(self, name: str, default: int) -> int | None:
        return (
            self.optional_int(name) if self.mapping.get(name) is not None else default
        )

    def optional_float(self, name: str) -> float | None:
        value = self.mapping.get(name)
        if value is None or not value.strip():
            return None
        return float(value)

    def optional_float_default(self, name: str, default: float) -> float | None:
        return (
            self.optional_float(name) if self.mapping.get(name) is not None else default
        )

    def optional_str(self, name: str) -> str | None:
        value = self.mapping.get(name)
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    def language_code(self, name: str, default: str = DEFAULT_LANGUAGE_CODE) -> str:
        value = self.mapping.get(name)
        if value is None or not value.strip():
            return default
        code = normalize_optional_iso_639_1_code(value)
        if code is None:
            raise ValueError(f"{name} must be an ISO 639-1 language code")
        return code

    def ratio_mapping(self, name: str, default: dict[str, float]) -> dict[str, float]:
        value = self.mapping.get(name)
        if value is None or not value.strip():
            return dict(default)
        normalized = value.strip()
        if normalized.startswith("{"):
            payload = json.loads(normalized)
            if not isinstance(payload, dict):
                raise ValueError(f"{name} must be a JSON object or key=value CSV")
            return {str(key): float(raw_value) for key, raw_value in payload.items()}
        ratios: dict[str, float] = {}
        for part in normalized.split(","):
            if not part.strip():
                continue
            key, separator, raw_value = part.partition("=")
            if not separator:
                raise ValueError(f"{name} entries must use key=value")
            ratios[key.strip()] = float(raw_value.strip())
        return ratios or dict(default)


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings for Atagia."""

    sqlite_path: str
    migrations_path: str
    manifests_path: str
    storage_backend: str
    redis_url: str
    openai_api_key: str | None
    openrouter_api_key: str | None
    openrouter_site_url: str
    openrouter_app_name: str
    llm_chat_model: str | None
    service_mode: bool
    service_api_key: str | None
    admin_api_key: str | None
    workers_enabled: bool
    debug: bool
    worker_circuit_breaker_enabled: bool = True
    worker_circuit_breaker_failure_threshold: int = 20
    worker_circuit_breaker_window_seconds: int = 180
    worker_circuit_breaker_min_failure_ratio: float = 0.8
    worker_transient_defer_seconds: float = 60.0
    worker_transient_defer_max_seconds: float = 300.0
    worker_transient_defer_max_count: int = 12
    worker_transient_defer_max_age_seconds: float = 3600.0
    worker_retry_backoff_initial_seconds: float = 1.0
    worker_retry_backoff_max_seconds: float = 30.0
    worker_dispatch_visibility_seconds: float = 30.0
    worker_dispatch_sweep_interval_seconds: float = 0.5
    worker_dispatch_batch_size: int = 100
    worker_execution_lease_seconds: float = 120.0
    worker_execution_heartbeat_seconds: float = 30.0
    worker_stream_reclaim_idle_seconds: float = 120.0
    service_process_count: int = 1
    llm_run_guard_enabled: bool = True
    llm_run_guard_mode: str = "enforce"
    # The process-wide run never ends, so an absolute lifetime cap on it is not
    # a health signal: a healthy long-lived process crosses any fixed number of
    # calls, failed calls, tokens or dollars eventually, and the guard then
    # blocks permanently until an operator runs the admin reset. Both absolute
    # caps therefore default to None. They remain settable for a deployment that
    # deliberately wants a finite process budget -- a violation of one is FINAL
    # by design, exactly because it is a budget and not a health reading.
    llm_run_guard_max_total_calls: int | None = None
    llm_run_guard_max_total_failed_calls: int | None = None
    # Health signals: bounded, self-clearing, and enforced over the recent
    # window below rather than over the process's whole life.
    llm_run_guard_max_failed_call_ratio: float | None = 0.50
    llm_run_guard_failed_ratio_min_calls: int = 20
    llm_run_guard_max_failed_calls_per_purpose: int | None = None
    llm_run_guard_max_failed_ratio_per_purpose: float | None = 0.50
    llm_run_guard_purpose_failure_ratio_min_calls: int = 10
    llm_run_guard_max_consecutive_failures_per_purpose: int | None = 8
    llm_run_guard_max_total_tokens: int | None = None
    llm_run_guard_max_reported_cost_usd: float | None = None
    # How many of the most recent calls a failure ratio is computed over, per
    # run and per purpose. A ratio over a run's whole life goes numb: after a
    # million healthy calls no live outage can move it, so the guard sleeps
    # through the failure it exists to catch. Shared by the runtime and bulk
    # guards -- the numbness argument applies to a 10k-call rebuild too.
    llm_run_guard_health_window_calls: int = 200
    # How long a health verdict blocks the process-wide run before it clears its
    # health window and lets traffic decide again. Blocking starves the counters
    # that would clear the verdict, so without an expiry the guard is a latch: a
    # provider having a bad hour wedges the process until someone notices.
    llm_run_guard_recovery_seconds: float = 60.0
    bulk_ingest_llm_run_guard_enabled: bool = True
    bulk_ingest_llm_run_guard_max_total_calls: int | None = 10000
    bulk_ingest_llm_run_guard_max_total_failed_calls: int | None = 40
    bulk_ingest_llm_run_guard_max_failed_call_ratio: float | None = 0.20
    bulk_ingest_llm_run_guard_failed_ratio_min_calls: int = 20
    bulk_ingest_llm_run_guard_max_failed_calls_per_purpose: int | None = None
    bulk_ingest_llm_run_guard_max_failed_ratio_per_purpose: float | None = 0.30
    bulk_ingest_llm_run_guard_purpose_failure_ratio_min_calls: int = 10
    bulk_ingest_llm_run_guard_max_consecutive_failures_per_purpose: int | None = 4
    bulk_ingest_llm_run_guard_max_total_tokens: int | None = None
    bulk_ingest_llm_run_guard_max_reported_cost_usd: float | None = None
    bulk_ingest_llm_run_guard_max_wall_time_seconds: float | None = 14400.0
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    kimi_api_key: str | None = None
    minimax_api_key: str | None = None
    typesafe_api_key: str | None = None
    anthropic_base_url: str | None = None
    anthropic_request_timeout_seconds: float = 120.0
    llm_request_timeout_seconds: float = 120.0
    # Matches the largest ordinary card fan-out while bounding combined workers.
    llm_max_concurrent_requests_per_provider: int = 4
    openai_base_url: str | None = None
    openai_embedding_base_url: str | None = None
    kimi_base_url: str | None = None
    minimax_base_url: str | None = None
    openrouter_base_url: str | None = None
    inference_access_mode: str = "unrestricted"
    local_llm_endpoints_file: str | None = None
    zero_cost_openrouter_profile: str | None = None
    llm_forced_global_model: str | None = None
    llm_ingest_model: str | None = None
    llm_retrieval_model: str | None = None
    llm_finite_decisions_enabled: bool = False
    llm_finite_decision_model: str | None = None
    llm_component_models: dict[str, str] = field(default_factory=dict)
    card_examples_enabled: bool = True
    llm_component_examples: dict[str, bool] = field(default_factory=dict)
    llm_intimacy_ingest_model: str | None = None
    llm_intimacy_retrieval_model: str | None = None
    llm_intimacy_component_models: dict[str, str] = field(default_factory=dict)
    llm_intimacy_proactive_routing_enabled: bool = False
    llm_structured_output_retry_attempts: int = 1
    llm_structured_output_rescue_enabled: bool = False
    llm_structured_output_rescue_model: str | None = (
        DEFAULT_STRUCTURED_OUTPUT_RESCUE_MODEL
    )
    llm_debug_io_enabled: bool = False
    llm_debug_io_dir: str = "./data/llm_debug"
    llm_debug_io_purposes: tuple[str, ...] = ()
    llm_debug_io_raw: bool = False
    llm_debug_io_max_chars: int = 50_000
    diagnostic_capture_enabled: bool = False
    diagnostic_capture_dir: str = "./data/diagnostic_captures"
    diagnostic_capture_max_blob_bytes: int = 1_048_576
    diagnostic_capture_max_session_bytes: int = 104_857_600
    operational_profiles_path: str = default_resource_path("operational_profiles")
    artifact_blob_storage_kind: str = "sqlite_blob"
    artifact_blob_storage_path: str = "./data/artifact_blobs"
    allow_admin_export_anonymization: bool = False
    allow_insecure_http: bool = False
    default_language_code: str = DEFAULT_LANGUAGE_CODE
    consequence_detector_card_concurrency: int = 2
    compactor_summary_card_concurrency: int = 4
    applicability_scorer_card_concurrency: int = 4
    embedding_backend: str = "none"
    embedding_model: str | None = None
    embedding_dimension: int = 1536
    embedding_vector_limit_cap: int = 50
    embedding_search_overfetch_multiplier: int = 4
    rrf_k: int = 60
    recall_recovery_scoring_max_candidates: int | None = None
    broad_list_comparable_rrf_enabled: bool = False
    fused_candidate_guard_ordering_enabled: bool = False
    memory_fts_canonical_bm25_weight: float = 1.2
    memory_fts_index_bm25_weight: float = 0.8
    lifecycle_decay_days: int = 7
    lifecycle_decay_rate: float = 0.9
    lifecycle_archive_vitality: float = 0.05
    lifecycle_archive_confidence: float = 0.3
    ephemeral_scoring_hours: int = 24
    lifecycle_ephemeral_ttl_hours: int = 24
    lifecycle_review_ttl_days: int = 7
    promotion_conv_to_ws_min_conversations: int = 2
    promotion_ws_to_global_min_sessions: int = 3
    promotion_require_mode_consistency: bool = True
    belief_tension_increment: float = 0.15
    belief_tension_decrement: float = 0.05
    belief_tension_threshold: float = 0.5
    skip_belief_revision: bool = False
    skip_compaction: bool = False
    episode_synthesis_max_episodes: int = 24
    opf_privacy_filter_enabled: bool = False
    opf_primary_url: str = "http://127.0.0.1:8008"
    opf_fallback_url: str = "http://127.0.0.1:8008"
    opf_timeout_seconds: float = 2.0
    privacy_validation_gate_enabled: bool = False
    privacy_validation_gate_timeout_seconds: float = 20.0
    privacy_validation_gate_max_source_chars: int = 6000
    privacy_validation_gate_max_summaries_gated_per_job: int = 50
    operational_high_risk_enabled: bool = False
    operational_allowed_profiles: tuple[str, ...] = ("normal", "low_power", "offline")
    context_cache_enabled: bool = True
    initial_context_package_read_enabled: bool = True
    initial_context_package_refresh_enabled: bool = True
    initial_context_package_curation_enabled: bool = False
    initial_context_package_prompt_max_tokens: int = 900
    initial_context_package_profile_max_tokens: int = 700
    initial_context_package_total_max_tokens: int = 2200
    initial_context_package_curated_block_max_tokens: int = 450
    initial_context_package_curated_max_items: int = 8
    initial_context_package_curation_max_output_tokens: int = 2048
    context_cache_min_ttl_seconds: int = 60
    context_cache_max_ttl_seconds: int = 3600
    temporary_default_ttl_seconds: int | None = None
    temporary_default_purge_on_close: bool = True
    tombstone_retention_days: int = 1825
    erasure_purge_streams: bool = True
    disable_chunking_extraction: bool = False
    chunking_extraction_threshold_tokens: int = 2048
    # Enables streamed extraction with a deterministic mechanical runaway guard.
    # When false, extraction uses the non-streamed structured-output path.
    extraction_watchdog_enabled: bool = True
    extraction_watchdog_allow_different_provider: bool = False
    extraction_watchdog_bounded_retry_max_items: int = 8
    extraction_watchdog_bounded_retry_max_output_tokens: int = 8192
    lifecycle_lazy_enabled: bool = True
    lifecycle_min_interval_seconds: int = 3600
    lifecycle_busy_timeout_ms: int = 1000
    lifecycle_busy_backoff_seconds: int = 60
    lifecycle_failure_backoff_seconds: int = 300
    lifecycle_worker_enabled: bool = False
    lifecycle_worker_interval_seconds: int = 3600
    retrieval_packets_dry_run_enabled: bool = False
    retrieval_packets_write_enabled: bool = False
    fact_facet_surfaces_enabled: bool = False
    fact_facet_retrieval_enabled: bool = False
    fact_facet_structured_only: bool = True
    fact_facet_span_coadmission_enabled: bool = True
    fact_facet_retrieval_limit: int = 12
    fact_facet_retrieval_rrf_weight: float = 1.1
    applicability_gate_mode: str = "off"
    small_corpus_token_threshold_ratio: float = 0.7
    assistant_guidance_enabled: bool = True
    response_mode: str = "normal"
    adaptive_retrieval: bool = True
    answer_stance: str = "reactive"
    answer_stance_prompt_variant: str = "baseline"
    answer_postcondition_guard_enabled: bool = False
    answer_postcondition_retry_max_output_tokens: int = 8192
    context_envelope_budget_tokens: int = 8192
    context_envelope_ratios: dict[str, float] = field(
        default_factory=lambda: dict(CONTEXT_ENVELOPE_DEFAULT_RATIOS)
    )
    benchmark_disable_raw_recent_transcript: bool = False
    recent_transcript_overage_ratio: float = 0.025
    topic_working_set_enabled: bool = True
    topic_working_set_update_mode: Literal["direct", "selective"] = "selective"
    topic_working_set_refresh_message_lag: int = 4
    topic_working_set_stale_message_lag: int = 10
    topic_working_set_refresh_token_lag: int = 2000
    topic_working_set_stale_token_lag: int = 5000
    topic_working_set_refresh_batch_messages: int = 8
    graph_projection_enabled: bool = False
    # FTS-backed verbatim evidence as first-class search channel.
    verbatim_evidence_search_enabled: bool = True
    # Evidence-search RRF weight, slightly lower than memory_objects by
    # default because memories are more focused and verbatim evidence is
    # a recall safety net. Tuning is deferred to Wave 2.
    verbatim_evidence_search_rrf_weight: float = 0.75
    # Maximum conversation windows fetched per sub-query from the
    # evidence-search channel before fusion. It is a secondary lane, so
    # this ceiling is modest.
    verbatim_evidence_search_limit: int = 8
    # Conversation window size (messages per window) used by the
    # evidence-search channel. Default 3 with 1-turn overlap keeps
    # neighbouring evidence accessible.
    verbatim_evidence_window_size: int = 3
    verbatim_evidence_window_overlap: int = 1
    openai_proxy_model_id: str = "atagia-memory-proxy"
    openai_proxy_upstream_model: str | None = None
    openai_proxy_default_mode: str | None = None
    openai_proxy_max_output_tokens: int = 8192
    request_max_body_bytes: int = 32 * 1024 * 1024
    request_max_message_text_bytes: int = 256 * 1024
    request_max_attachments: int = 16
    request_max_attachment_decoded_bytes: int = 10 * 1024 * 1024
    request_max_attachments_decoded_bytes: int = 20 * 1024 * 1024
    request_max_metadata_bytes: int = 64 * 1024
    cors_allowed_origins: tuple[str, ...] = ()
    llm_technical_recovery_enabled: bool = True
    llm_output_limit_retry_attempts: int = 1
    llm_runaway_watchdog_enabled: bool = True
    llm_runaway_min_elapsed_seconds: float = 8.0
    llm_runaway_min_output_tokens: int = 2048
    llm_runaway_check_interval_tokens: int = 1024
    llm_runaway_max_checks: int = 2
    llm_runaway_hard_abort_min_output_tokens: int = 4096
    llm_runaway_min_repeat_count: int = 3
    llm_runaway_min_repeat_ratio_tokens: float = 0.12
    llm_runaway_output_input_ratio: float = 12.0
    llm_runaway_hard_output_input_ratio: float = 8.0

    def __post_init__(self) -> None:
        validate_finite_decision_configuration(self)
        if self.inference_access_mode not in {
            "unrestricted",
            "local_only",
            "zero_cost",
        }:
            raise ValueError(
                "inference_access_mode must be one of: unrestricted, local_only, zero_cost"
            )
        for field_name in (
            "local_llm_endpoints_file",
            "zero_cost_openrouter_profile",
        ):
            value = getattr(self, field_name)
            if value is not None and (not value.strip() or value != value.strip()):
                raise ValueError(
                    f"{field_name} must be a non-empty string without surrounding whitespace when set"
                )
        if not self.openai_proxy_model_id.strip():
            raise ValueError("openai_proxy_model_id cannot be blank")
        for field_name in (
            "openai_proxy_max_output_tokens",
            "request_max_body_bytes",
            "request_max_message_text_bytes",
            "request_max_attachments",
            "request_max_attachment_decoded_bytes",
            "request_max_attachments_decoded_bytes",
            "request_max_metadata_bytes",
        ):
            if int(getattr(self, field_name)) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if (
            self.request_max_attachments_decoded_bytes
            < self.request_max_attachment_decoded_bytes
        ):
            raise ValueError(
                "request_max_attachments_decoded_bytes must be >= "
                "request_max_attachment_decoded_bytes"
            )
        if self.context_cache_min_ttl_seconds <= 0:
            raise ValueError("context_cache_min_ttl_seconds must be positive")
        if self.worker_circuit_breaker_failure_threshold <= 0:
            raise ValueError(
                "worker_circuit_breaker_failure_threshold must be positive"
            )
        if self.worker_circuit_breaker_window_seconds <= 0:
            raise ValueError("worker_circuit_breaker_window_seconds must be positive")
        if not 0.0 <= self.worker_circuit_breaker_min_failure_ratio <= 1.0:
            raise ValueError(
                "worker_circuit_breaker_min_failure_ratio must be in the interval [0.0, 1.0]"
            )
        if self.worker_transient_defer_seconds <= 0:
            raise ValueError("worker_transient_defer_seconds must be positive")
        if self.worker_transient_defer_max_seconds <= 0:
            raise ValueError("worker_transient_defer_max_seconds must be positive")
        if self.worker_transient_defer_max_seconds < self.worker_transient_defer_seconds:
            raise ValueError(
                "worker_transient_defer_max_seconds must be >= worker_transient_defer_seconds"
            )
        if self.worker_transient_defer_max_count <= 0:
            raise ValueError("worker_transient_defer_max_count must be positive")
        if self.worker_transient_defer_max_age_seconds <= 0:
            raise ValueError("worker_transient_defer_max_age_seconds must be positive")
        if self.worker_retry_backoff_initial_seconds <= 0:
            raise ValueError("worker_retry_backoff_initial_seconds must be positive")
        if self.worker_retry_backoff_max_seconds < self.worker_retry_backoff_initial_seconds:
            raise ValueError(
                "worker_retry_backoff_max_seconds must be >= worker_retry_backoff_initial_seconds"
            )
        if self.worker_dispatch_visibility_seconds <= 0:
            raise ValueError("worker_dispatch_visibility_seconds must be positive")
        if self.worker_dispatch_sweep_interval_seconds <= 0:
            raise ValueError("worker_dispatch_sweep_interval_seconds must be positive")
        if self.worker_dispatch_batch_size <= 0:
            raise ValueError("worker_dispatch_batch_size must be positive")
        if self.worker_execution_lease_seconds <= 0:
            raise ValueError("worker_execution_lease_seconds must be positive")
        if self.worker_execution_heartbeat_seconds <= 0:
            raise ValueError("worker_execution_heartbeat_seconds must be positive")
        if self.worker_execution_heartbeat_seconds >= self.worker_execution_lease_seconds:
            raise ValueError(
                "worker_execution_heartbeat_seconds must be shorter than worker_execution_lease_seconds"
            )
        if self.worker_stream_reclaim_idle_seconds < self.worker_dispatch_visibility_seconds:
            raise ValueError(
                "worker_stream_reclaim_idle_seconds must be >= worker_dispatch_visibility_seconds"
            )
        if self.service_process_count <= 0:
            raise ValueError("service_process_count must be positive")
        if self.storage_backend == "inprocess" and self.service_process_count != 1:
            raise ValueError(
                "storage_backend=inprocess supports exactly one service process; use Redis for multi-process deployments"
            )
        if self.llm_run_guard_mode not in {"off", "audit", "enforce"}:
            raise ValueError("llm_run_guard_mode must be one of: off, audit, enforce")
        for field_name in (
            "llm_run_guard_max_total_calls",
            "llm_run_guard_max_total_failed_calls",
            "llm_run_guard_max_failed_calls_per_purpose",
            "llm_run_guard_max_consecutive_failures_per_purpose",
            "llm_run_guard_max_total_tokens",
            "bulk_ingest_llm_run_guard_max_total_calls",
            "bulk_ingest_llm_run_guard_max_total_failed_calls",
            "bulk_ingest_llm_run_guard_max_failed_calls_per_purpose",
            "bulk_ingest_llm_run_guard_max_consecutive_failures_per_purpose",
            "bulk_ingest_llm_run_guard_max_total_tokens",
        ):
            value = getattr(self, field_name)
            if value is not None and value <= 0:
                raise ValueError(f"{field_name} must be positive when set")
        for field_name in (
            "llm_run_guard_failed_ratio_min_calls",
            "llm_run_guard_purpose_failure_ratio_min_calls",
            "bulk_ingest_llm_run_guard_failed_ratio_min_calls",
            "bulk_ingest_llm_run_guard_purpose_failure_ratio_min_calls",
            "llm_run_guard_health_window_calls",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.llm_run_guard_recovery_seconds <= 0.0:
            raise ValueError("llm_run_guard_recovery_seconds must be positive")
        for field_name in (
            "llm_run_guard_max_failed_call_ratio",
            "llm_run_guard_max_failed_ratio_per_purpose",
            "bulk_ingest_llm_run_guard_max_failed_call_ratio",
            "bulk_ingest_llm_run_guard_max_failed_ratio_per_purpose",
        ):
            value = getattr(self, field_name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in the interval [0.0, 1.0]")
        for field_name in (
            "llm_run_guard_max_reported_cost_usd",
            "bulk_ingest_llm_run_guard_max_reported_cost_usd",
            "bulk_ingest_llm_run_guard_max_wall_time_seconds",
        ):
            value = getattr(self, field_name)
            if value is not None and value <= 0.0:
                raise ValueError(f"{field_name} must be positive when set")
        for field_name in (
            "llm_runaway_min_output_tokens",
            "llm_runaway_check_interval_tokens",
            "llm_runaway_hard_abort_min_output_tokens",
            "llm_runaway_min_repeat_count",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.llm_output_limit_retry_attempts < 0:
            raise ValueError("llm_output_limit_retry_attempts must be non-negative")
        if self.llm_runaway_min_elapsed_seconds < 0:
            raise ValueError("llm_runaway_min_elapsed_seconds must be non-negative")
        if self.llm_runaway_max_checks < 0:
            raise ValueError("llm_runaway_max_checks must be non-negative")
        if not 0.0 < self.llm_runaway_min_repeat_ratio_tokens <= 1.0:
            raise ValueError(
                "llm_runaway_min_repeat_ratio_tokens must be in the interval (0.0, 1.0]"
            )
        for field_name in (
            "llm_runaway_output_input_ratio",
            "llm_runaway_hard_output_input_ratio",
        ):
            if getattr(self, field_name) <= 0.0:
                raise ValueError(f"{field_name} must be positive")
        if self.context_cache_max_ttl_seconds <= 0:
            raise ValueError("context_cache_max_ttl_seconds must be positive")
        if self.context_cache_max_ttl_seconds < self.context_cache_min_ttl_seconds:
            raise ValueError(
                "context_cache_max_ttl_seconds must be >= context_cache_min_ttl_seconds"
            )
        if self.initial_context_package_prompt_max_tokens <= 0:
            raise ValueError(
                "initial_context_package_prompt_max_tokens must be positive"
            )
        if self.initial_context_package_profile_max_tokens <= 0:
            raise ValueError(
                "initial_context_package_profile_max_tokens must be positive"
            )
        if self.initial_context_package_total_max_tokens <= 0:
            raise ValueError(
                "initial_context_package_total_max_tokens must be positive"
            )
        if self.initial_context_package_curated_block_max_tokens <= 0:
            raise ValueError(
                "initial_context_package_curated_block_max_tokens must be positive"
            )
        if self.initial_context_package_curated_max_items <= 0:
            raise ValueError(
                "initial_context_package_curated_max_items must be positive"
            )
        if self.initial_context_package_curation_max_output_tokens <= 0:
            raise ValueError(
                "initial_context_package_curation_max_output_tokens must be positive"
            )
        if self.consequence_detector_card_concurrency <= 0:
            raise ValueError("consequence_detector_card_concurrency must be positive")
        if self.compactor_summary_card_concurrency <= 0:
            raise ValueError("compactor_summary_card_concurrency must be positive")
        if self.applicability_scorer_card_concurrency <= 0:
            raise ValueError("applicability_scorer_card_concurrency must be positive")
        if self.llm_max_concurrent_requests_per_provider <= 0:
            raise ValueError("llm_max_concurrent_requests_per_provider must be positive")
        if (
            self.temporary_default_ttl_seconds is not None
            and self.temporary_default_ttl_seconds <= 0
        ):
            raise ValueError("temporary_default_ttl_seconds must be positive when set")
        if self.tombstone_retention_days <= 0:
            raise ValueError("tombstone_retention_days must be positive")
        if self.anthropic_request_timeout_seconds <= 0:
            raise ValueError("anthropic_request_timeout_seconds must be positive")
        if self.llm_request_timeout_seconds <= 0:
            raise ValueError("llm_request_timeout_seconds must be positive")
        if self.chunking_extraction_threshold_tokens <= 0:
            raise ValueError("chunking_extraction_threshold_tokens must be positive")
        if self.extraction_watchdog_bounded_retry_max_items <= 0:
            raise ValueError(
                "extraction_watchdog_bounded_retry_max_items must be positive"
            )
        if self.extraction_watchdog_bounded_retry_max_output_tokens < 8192:
            raise ValueError(
                "extraction_watchdog_bounded_retry_max_output_tokens must be at least 8192"
            )
        if self.answer_postcondition_retry_max_output_tokens < 8192:
            raise ValueError(
                "answer_postcondition_retry_max_output_tokens must be at least 8192"
            )
        if self.response_mode not in RESPONSE_MODES:
            raise ValueError(
                "response_mode must be one of: normal, fast, smart_fast"
            )
        if self.answer_stance not in ANSWER_STANCES:
            raise ValueError(
                "answer_stance must be one of: reactive, proactive"
            )
        if self.answer_stance_prompt_variant not in ANSWER_STANCE_PROMPT_VARIANTS:
            raise ValueError(
                "answer_stance_prompt_variant must be one of: baseline, "
                "template_v1, template_v2, template_v3, category_hint, "
                "exact_template, no_substitute, binary_gate, if_else_gate, "
                "label_precision, strict_adjacent_silence, liability_guard, "
                "hard_exact_stop, hard_no_extra, partial_evidence_override, "
                "stance_gate_combo"
            )
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if (
            self.recall_recovery_scoring_max_candidates is not None
            and self.recall_recovery_scoring_max_candidates <= 0
        ):
            raise ValueError(
                "recall_recovery_scoring_max_candidates must be positive when set"
            )
        if self.embedding_vector_limit_cap <= 0:
            raise ValueError("embedding_vector_limit_cap must be positive")
        if self.embedding_search_overfetch_multiplier <= 0:
            raise ValueError("embedding_search_overfetch_multiplier must be positive")
        if (
            not math.isfinite(self.memory_fts_canonical_bm25_weight)
            or self.memory_fts_canonical_bm25_weight <= 0.0
        ):
            raise ValueError(
                "memory_fts_canonical_bm25_weight must be a finite positive number"
            )
        if (
            not math.isfinite(self.memory_fts_index_bm25_weight)
            or self.memory_fts_index_bm25_weight <= 0.0
        ):
            raise ValueError(
                "memory_fts_index_bm25_weight must be a finite positive number"
            )
        if self.lifecycle_min_interval_seconds <= 0:
            raise ValueError("lifecycle_min_interval_seconds must be positive")
        if self.lifecycle_busy_timeout_ms <= 0:
            raise ValueError("lifecycle_busy_timeout_ms must be positive")
        if self.lifecycle_busy_backoff_seconds <= 0:
            raise ValueError("lifecycle_busy_backoff_seconds must be positive")
        if self.lifecycle_failure_backoff_seconds <= 0:
            raise ValueError("lifecycle_failure_backoff_seconds must be positive")
        if (
            self.lifecycle_worker_enabled
            and self.lifecycle_worker_interval_seconds <= 0
        ):
            raise ValueError("lifecycle_worker_interval_seconds must be positive")
        if self.belief_tension_increment <= 0:
            raise ValueError("belief_tension_increment must be positive")
        if self.belief_tension_decrement <= 0:
            raise ValueError("belief_tension_decrement must be positive")
        if self.belief_tension_threshold < 0:
            raise ValueError("belief_tension_threshold must be non-negative")
        if self.episode_synthesis_max_episodes <= 0:
            raise ValueError("episode_synthesis_max_episodes must be positive")
        if self.ephemeral_scoring_hours <= 0:
            raise ValueError("ephemeral_scoring_hours must be positive")
        if self.opf_timeout_seconds <= 0:
            raise ValueError("opf_timeout_seconds must be positive")
        if self.privacy_validation_gate_timeout_seconds <= 0:
            raise ValueError("privacy_validation_gate_timeout_seconds must be positive")
        if self.privacy_validation_gate_max_source_chars <= 0:
            raise ValueError(
                "privacy_validation_gate_max_source_chars must be positive"
            )
        if self.privacy_validation_gate_max_summaries_gated_per_job < 0:
            raise ValueError(
                "privacy_validation_gate_max_summaries_gated_per_job must be non-negative"
            )
        if not self.operational_allowed_profiles:
            raise ValueError(
                "operational_allowed_profiles must contain at least one profile"
            )
        if any(
            not profile_id.strip() for profile_id in self.operational_allowed_profiles
        ):
            raise ValueError(
                "operational_allowed_profiles cannot contain blank profile ids"
            )
        if self.llm_structured_output_retry_attempts < 0:
            raise ValueError(
                "llm_structured_output_retry_attempts must be non-negative"
            )
        if (
            self.llm_structured_output_rescue_enabled
            and not (self.llm_structured_output_rescue_model or "").strip()
        ):
            raise ValueError(
                "llm_structured_output_rescue_model is required when structured-output rescue is enabled"
            )
        if not self.llm_debug_io_dir.strip():
            raise ValueError("llm_debug_io_dir cannot be blank")
        if self.llm_debug_io_max_chars < 0:
            raise ValueError("llm_debug_io_max_chars must be non-negative")
        if self.artifact_blob_storage_kind not in {"sqlite_blob", "local_file"}:
            raise ValueError(
                "artifact_blob_storage_kind must be 'sqlite_blob'"
            )
        if (
            self.artifact_blob_storage_kind == "local_file"
            and not self.artifact_blob_storage_path.strip()
        ):
            raise ValueError(
                "artifact_blob_storage_path is required to migrate legacy local_file artifacts"
            )
        if not 0.0 <= self.small_corpus_token_threshold_ratio <= 1.0:
            raise ValueError(
                "small_corpus_token_threshold_ratio must be in the interval [0.0, 1.0]"
            )
        allocate_context_envelope_budget(
            self.context_envelope_budget_tokens,
            self.context_envelope_ratios,
        )
        if not 0.0 <= self.recent_transcript_overage_ratio <= 1.0:
            raise ValueError(
                "recent_transcript_overage_ratio must be in the interval [0.0, 1.0]"
            )
        if self.topic_working_set_update_mode not in {"direct", "selective"}:
            raise ValueError("topic_working_set_update_mode must be direct or selective")
        if self.topic_working_set_refresh_message_lag <= 0:
            raise ValueError("topic_working_set_refresh_message_lag must be positive")
        if (
            self.topic_working_set_stale_message_lag
            < self.topic_working_set_refresh_message_lag
        ):
            raise ValueError(
                "topic_working_set_stale_message_lag must be >= topic_working_set_refresh_message_lag"
            )
        if self.topic_working_set_refresh_token_lag <= 0:
            raise ValueError("topic_working_set_refresh_token_lag must be positive")
        if (
            self.topic_working_set_stale_token_lag
            < self.topic_working_set_refresh_token_lag
        ):
            raise ValueError(
                "topic_working_set_stale_token_lag must be >= topic_working_set_refresh_token_lag"
            )
        if self.topic_working_set_refresh_batch_messages <= 0:
            raise ValueError(
                "topic_working_set_refresh_batch_messages must be positive"
            )
        if not 0.0 <= self.verbatim_evidence_search_rrf_weight <= 2.0:
            raise ValueError(
                "verbatim_evidence_search_rrf_weight must be in the interval [0.0, 2.0]"
            )
        if self.verbatim_evidence_search_limit < 0:
            raise ValueError("verbatim_evidence_search_limit must be non-negative")
        if self.fact_facet_retrieval_limit < 0:
            raise ValueError("fact_facet_retrieval_limit must be non-negative")
        if not 0.0 <= self.fact_facet_retrieval_rrf_weight <= 2.0:
            raise ValueError(
                "fact_facet_retrieval_rrf_weight must be in the interval [0.0, 2.0]"
            )
        if self.applicability_gate_mode not in {"off", "shadow", "enforced"}:
            raise ValueError(
                "applicability_gate_mode must be one of: off, shadow, enforced"
            )
        if (
            self.verbatim_evidence_window_size < 2
            or self.verbatim_evidence_window_size > 4
        ):
            raise ValueError("verbatim_evidence_window_size must be between 2 and 4")
        if self.verbatim_evidence_window_overlap < 0:
            raise ValueError("verbatim_evidence_window_overlap must be non-negative")
        if self.verbatim_evidence_window_overlap >= self.verbatim_evidence_window_size:
            raise ValueError(
                "verbatim_evidence_window_overlap must be strictly less than verbatim_evidence_window_size"
            )

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        """Build settings from an environment mapping.

        ``environ`` defaults to the process environment, and only that path
        loads ``.env``: an explicit mapping is read exactly as given so a
        caller can resolve a baseline (or a synthetic environment) without
        touching or depending on process-global state.
        """
        if environ is None:
            _load_dotenv_once()
            environ = os.environ
        env = _EnvReader(environ)
        return cls(
            sqlite_path=env.get("ATAGIA_SQLITE_PATH", "./data/atagia.db"),
            migrations_path=configured_resource_path(
                "migrations",
                env.get("ATAGIA_MIGRATIONS_PATH"),
            ),
            manifests_path=configured_resource_path(
                "manifests",
                env.get("ATAGIA_MANIFESTS_PATH"),
            ),
            operational_profiles_path=configured_resource_path(
                "operational_profiles",
                env.get("ATAGIA_OPERATIONAL_PROFILES_PATH"),
            ),
            artifact_blob_storage_kind=env.get(
                "ATAGIA_ARTIFACT_BLOB_STORAGE_KIND",
                "sqlite_blob",
            )
            .strip()
            .lower(),
            artifact_blob_storage_path=env.get(
                "ATAGIA_ARTIFACT_BLOB_STORAGE_PATH",
                "./data/artifact_blobs",
            ),
            storage_backend=env.get("ATAGIA_STORAGE_BACKEND", "inprocess")
            .strip()
            .lower(),
            redis_url=env.get("ATAGIA_REDIS_URL", "redis://localhost:6379/0"),
            anthropic_api_key=env.get("ATAGIA_ANTHROPIC_API_KEY") or None,
            openai_api_key=env.get("ATAGIA_OPENAI_API_KEY") or None,
            openrouter_api_key=env.get("ATAGIA_OPENROUTER_API_KEY") or None,
            kimi_api_key=env.get("ATAGIA_KIMI_API_KEY") or None,
            minimax_api_key=env.get("ATAGIA_MINIMAX_API_KEY") or None,
            typesafe_api_key=env.get("ATAGIA_TYPESAFE_API_KEY") or None,
            anthropic_base_url=env.get("ATAGIA_ANTHROPIC_BASE_URL") or None,
            anthropic_request_timeout_seconds=float(
                env.get("ATAGIA_ANTHROPIC_REQUEST_TIMEOUT_SECONDS", "120.0")
            ),
            llm_request_timeout_seconds=float(
                env.get("ATAGIA_LLM_REQUEST_TIMEOUT_SECONDS", "120.0")
            ),
            llm_max_concurrent_requests_per_provider=int(
                env.get("ATAGIA_LLM_MAX_CONCURRENT_REQUESTS_PER_PROVIDER", "4")
            ),
            openai_base_url=env.get("ATAGIA_OPENAI_BASE_URL") or None,
            openai_embedding_base_url=(
                env.get("ATAGIA_OPENAI_EMBEDDING_BASE_URL") or None
            ),
            kimi_base_url=env.get("ATAGIA_KIMI_BASE_URL") or None,
            minimax_base_url=env.get("ATAGIA_MINIMAX_BASE_URL") or None,
            openrouter_base_url=env.get("ATAGIA_OPENROUTER_BASE_URL") or None,
            inference_access_mode=env.get(
                "ATAGIA_INFERENCE_ACCESS_MODE", "unrestricted"
            )
            .strip()
            .lower(),
            local_llm_endpoints_file=env.optional_str(
                "ATAGIA_LOCAL_LLM_ENDPOINTS_FILE"
            ),
            zero_cost_openrouter_profile=env.optional_str(
                "ATAGIA_ZERO_COST_OPENROUTER_PROFILE"
            ),
            openrouter_site_url=env.get(
                "ATAGIA_OPENROUTER_SITE_URL", "http://localhost"
            ),
            openrouter_app_name=env.get("ATAGIA_OPENROUTER_APP_NAME", "Atagia"),
            llm_chat_model=env.get("ATAGIA_LLM_CHAT_MODEL") or None,
            llm_forced_global_model=env.get("ATAGIA_LLM_FORCED_GLOBAL_MODEL") or None,
            llm_ingest_model=env.get("ATAGIA_LLM_INGEST_MODEL") or None,
            llm_retrieval_model=env.get("ATAGIA_LLM_RETRIEVAL_MODEL") or None,
            llm_finite_decisions_enabled=env.bool(
                "ATAGIA_LLM_FINITE_DECISIONS_ENABLED",
                False,
            ),
            llm_finite_decision_model=env.get(
                "ATAGIA_LLM_FINITE_DECISION_MODEL"
            )
            or None,
            llm_component_models=component_env_models_from_env(env.mapping),
            card_examples_enabled=env.bool("ATAGIA_CARD_EXAMPLES_ENABLED", True),
            llm_component_examples=component_env_examples_from_env(env.mapping),
            llm_intimacy_ingest_model=env.get("ATAGIA_LLM_INTIMACY_INGEST_MODEL")
            or None,
            llm_intimacy_retrieval_model=(
                env.get("ATAGIA_LLM_INTIMACY_RETRIEVAL_MODEL") or None
            ),
            llm_intimacy_component_models=intimacy_component_env_models_from_env(
                env.mapping
            ),
            llm_intimacy_proactive_routing_enabled=env.bool(
                "ATAGIA_LLM_INTIMACY_PROACTIVE_ROUTING_ENABLED",
                False,
            ),
            llm_structured_output_retry_attempts=int(
                env.get("ATAGIA_LLM_STRUCTURED_OUTPUT_RETRY_ATTEMPTS", "1")
            ),
            llm_structured_output_rescue_enabled=env.bool(
                "ATAGIA_LLM_STRUCTURED_OUTPUT_RESCUE_ENABLED",
                False,
            ),
            llm_structured_output_rescue_model=(
                env.optional_str("ATAGIA_LLM_STRUCTURED_OUTPUT_RESCUE_MODEL")
                or DEFAULT_STRUCTURED_OUTPUT_RESCUE_MODEL
            ),
            llm_technical_recovery_enabled=env.bool(
                "ATAGIA_LLM_TECHNICAL_RECOVERY_ENABLED",
                True,
            ),
            llm_output_limit_retry_attempts=int(
                env.get("ATAGIA_LLM_OUTPUT_LIMIT_RETRY_ATTEMPTS", "1")
            ),
            llm_runaway_watchdog_enabled=env.bool(
                "ATAGIA_LLM_RUNAWAY_WATCHDOG_ENABLED",
                True,
            ),
            llm_runaway_min_elapsed_seconds=float(
                env.get("ATAGIA_LLM_RUNAWAY_MIN_ELAPSED_SECONDS", "8.0")
            ),
            llm_runaway_min_output_tokens=int(
                env.get("ATAGIA_LLM_RUNAWAY_MIN_OUTPUT_TOKENS", "2048")
            ),
            llm_runaway_check_interval_tokens=int(
                env.get("ATAGIA_LLM_RUNAWAY_CHECK_INTERVAL_TOKENS", "1024")
            ),
            llm_runaway_max_checks=int(
                env.get("ATAGIA_LLM_RUNAWAY_MAX_CHECKS", "2")
            ),
            llm_runaway_hard_abort_min_output_tokens=int(
                env.get("ATAGIA_LLM_RUNAWAY_HARD_ABORT_MIN_OUTPUT_TOKENS", "4096")
            ),
            llm_runaway_min_repeat_count=int(
                env.get("ATAGIA_LLM_RUNAWAY_MIN_REPEAT_COUNT", "3")
            ),
            llm_runaway_min_repeat_ratio_tokens=float(
                env.get("ATAGIA_LLM_RUNAWAY_MIN_REPEAT_RATIO_TOKENS", "0.12")
            ),
            llm_runaway_output_input_ratio=float(
                env.get("ATAGIA_LLM_RUNAWAY_OUTPUT_INPUT_RATIO", "12.0")
            ),
            llm_runaway_hard_output_input_ratio=float(
                env.get("ATAGIA_LLM_RUNAWAY_HARD_OUTPUT_INPUT_RATIO", "8.0")
            ),
            llm_debug_io_enabled=env.bool("ATAGIA_DEBUG_LLM_IO", False),
            diagnostic_capture_enabled=env.bool("ATAGIA_DIAGNOSTIC_CAPTURE_ENABLED", False),
            diagnostic_capture_dir=env.get("ATAGIA_DIAGNOSTIC_CAPTURE_DIR", "./data/diagnostic_captures"),
            diagnostic_capture_max_blob_bytes=int(env.get("ATAGIA_DIAGNOSTIC_CAPTURE_MAX_BLOB_BYTES", "1048576")),
            diagnostic_capture_max_session_bytes=int(env.get("ATAGIA_DIAGNOSTIC_CAPTURE_MAX_SESSION_BYTES", "104857600")),
            llm_debug_io_dir=env.get(
                "ATAGIA_DEBUG_LLM_IO_DIR",
                "./data/llm_debug",
            ),
            llm_debug_io_purposes=env.csv_tuple("ATAGIA_DEBUG_LLM_IO_PURPOSES", ()),
            llm_debug_io_raw=env.bool("ATAGIA_DEBUG_LLM_IO_RAW", False),
            llm_debug_io_max_chars=int(
                env.get("ATAGIA_DEBUG_LLM_IO_MAX_CHARS", "50000")
            ),
            service_mode=env.bool("ATAGIA_SERVICE_MODE", False),
            service_api_key=env.get("ATAGIA_SERVICE_API_KEY") or None,
            admin_api_key=env.get("ATAGIA_ADMIN_API_KEY") or None,
            allow_admin_export_anonymization=env.bool(
                "ATAGIA_ALLOW_ADMIN_EXPORT_ANONYMIZATION",
                False,
            ),
            workers_enabled=env.bool("ATAGIA_WORKERS_ENABLED", False),
            debug=env.bool("ATAGIA_DEBUG", False),
            worker_circuit_breaker_enabled=env.bool(
                "ATAGIA_WORKER_CIRCUIT_BREAKER_ENABLED",
                True,
            ),
            worker_circuit_breaker_failure_threshold=int(
                env.get("ATAGIA_WORKER_CIRCUIT_BREAKER_FAILURE_THRESHOLD", "20")
            ),
            worker_circuit_breaker_window_seconds=int(
                env.get("ATAGIA_WORKER_CIRCUIT_BREAKER_WINDOW_SECONDS", "180")
            ),
            worker_circuit_breaker_min_failure_ratio=float(
                env.get("ATAGIA_WORKER_CIRCUIT_BREAKER_MIN_FAILURE_RATIO", "0.8")
            ),
            worker_transient_defer_seconds=float(
                env.get("ATAGIA_WORKER_TRANSIENT_DEFER_SECONDS", "60.0")
            ),
            worker_transient_defer_max_seconds=float(
                env.get("ATAGIA_WORKER_TRANSIENT_DEFER_MAX_SECONDS", "300.0")
            ),
            worker_transient_defer_max_count=int(
                env.get("ATAGIA_WORKER_TRANSIENT_DEFER_MAX_COUNT", "12")
            ),
            worker_transient_defer_max_age_seconds=float(
                env.get("ATAGIA_WORKER_TRANSIENT_DEFER_MAX_AGE_SECONDS", "3600.0")
            ),
            worker_retry_backoff_initial_seconds=float(
                env.get("ATAGIA_WORKER_RETRY_BACKOFF_INITIAL_SECONDS", "1.0")
            ),
            worker_retry_backoff_max_seconds=float(
                env.get("ATAGIA_WORKER_RETRY_BACKOFF_MAX_SECONDS", "30.0")
            ),
            worker_dispatch_visibility_seconds=float(
                env.get("ATAGIA_WORKER_DISPATCH_VISIBILITY_SECONDS", "30.0")
            ),
            worker_dispatch_sweep_interval_seconds=float(
                env.get("ATAGIA_WORKER_DISPATCH_SWEEP_INTERVAL_SECONDS", "0.5")
            ),
            worker_dispatch_batch_size=int(
                env.get("ATAGIA_WORKER_DISPATCH_BATCH_SIZE", "100")
            ),
            worker_execution_lease_seconds=float(
                env.get("ATAGIA_WORKER_EXECUTION_LEASE_SECONDS", "120.0")
            ),
            worker_execution_heartbeat_seconds=float(
                env.get("ATAGIA_WORKER_EXECUTION_HEARTBEAT_SECONDS", "30.0")
            ),
            worker_stream_reclaim_idle_seconds=float(
                env.get("ATAGIA_WORKER_STREAM_RECLAIM_IDLE_SECONDS", "120.0")
            ),
            service_process_count=int(
                env.get(
                    "ATAGIA_SERVICE_PROCESS_COUNT",
                    env.get("WEB_CONCURRENCY", "1"),
                )
            ),
            llm_run_guard_enabled=env.bool("ATAGIA_LLM_RUN_GUARD_ENABLED", True),
            llm_run_guard_mode=env.get("ATAGIA_LLM_RUN_GUARD_MODE", "enforce")
            .strip()
            .lower(),
            llm_run_guard_max_total_calls=env.optional_int(
                "ATAGIA_LLM_RUN_GUARD_MAX_TOTAL_CALLS"
            ),
            llm_run_guard_max_total_failed_calls=env.optional_int(
                "ATAGIA_LLM_RUN_GUARD_MAX_TOTAL_FAILED_CALLS"
            ),
            llm_run_guard_max_failed_call_ratio=env.optional_float_default(
                "ATAGIA_LLM_RUN_GUARD_MAX_FAILED_CALL_RATIO",
                0.50,
            ),
            llm_run_guard_failed_ratio_min_calls=int(
                env.get("ATAGIA_LLM_RUN_GUARD_FAILED_RATIO_MIN_CALLS", "20")
            ),
            llm_run_guard_max_failed_calls_per_purpose=env.optional_int(
                "ATAGIA_LLM_RUN_GUARD_MAX_FAILED_CALLS_PER_PURPOSE"
            ),
            llm_run_guard_max_failed_ratio_per_purpose=env.optional_float_default(
                "ATAGIA_LLM_RUN_GUARD_MAX_FAILED_RATIO_PER_PURPOSE",
                0.50,
            ),
            llm_run_guard_purpose_failure_ratio_min_calls=int(
                env.get(
                    "ATAGIA_LLM_RUN_GUARD_PURPOSE_FAILURE_RATIO_MIN_CALLS",
                    "10",
                )
            ),
            llm_run_guard_max_consecutive_failures_per_purpose=(
                env.optional_int_default(
                    "ATAGIA_LLM_RUN_GUARD_MAX_CONSECUTIVE_FAILURES_PER_PURPOSE",
                    8,
                )
            ),
            llm_run_guard_max_total_tokens=env.optional_int(
                "ATAGIA_LLM_RUN_GUARD_MAX_TOTAL_TOKENS"
            ),
            llm_run_guard_max_reported_cost_usd=env.optional_float(
                "ATAGIA_LLM_RUN_GUARD_MAX_REPORTED_COST_USD"
            ),
            llm_run_guard_health_window_calls=int(
                env.get("ATAGIA_LLM_RUN_GUARD_HEALTH_WINDOW_CALLS", "200")
            ),
            llm_run_guard_recovery_seconds=float(
                env.get("ATAGIA_LLM_RUN_GUARD_RECOVERY_SECONDS", "60")
            ),
            bulk_ingest_llm_run_guard_enabled=env.bool(
                "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_ENABLED",
                True,
            ),
            bulk_ingest_llm_run_guard_max_total_calls=env.optional_int_default(
                "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_MAX_TOTAL_CALLS",
                10000,
            ),
            bulk_ingest_llm_run_guard_max_total_failed_calls=(
                env.optional_int_default(
                    "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_MAX_TOTAL_FAILED_CALLS",
                    40,
                )
            ),
            bulk_ingest_llm_run_guard_max_failed_call_ratio=(
                env.optional_float_default(
                    "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_MAX_FAILED_CALL_RATIO",
                    0.20,
                )
            ),
            bulk_ingest_llm_run_guard_failed_ratio_min_calls=int(
                env.get(
                    "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_FAILED_RATIO_MIN_CALLS",
                    "20",
                )
            ),
            bulk_ingest_llm_run_guard_max_failed_calls_per_purpose=env.optional_int(
                "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_MAX_FAILED_CALLS_PER_PURPOSE"
            ),
            bulk_ingest_llm_run_guard_max_failed_ratio_per_purpose=(
                env.optional_float_default(
                    "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_MAX_FAILED_RATIO_PER_PURPOSE",
                    0.30,
                )
            ),
            bulk_ingest_llm_run_guard_purpose_failure_ratio_min_calls=int(
                env.get(
                    "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_PURPOSE_FAILURE_RATIO_MIN_CALLS",
                    "10",
                )
            ),
            bulk_ingest_llm_run_guard_max_consecutive_failures_per_purpose=(
                env.optional_int_default(
                    "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_MAX_CONSECUTIVE_FAILURES_PER_PURPOSE",
                    4,
                )
            ),
            bulk_ingest_llm_run_guard_max_total_tokens=env.optional_int(
                "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_MAX_TOTAL_TOKENS"
            ),
            bulk_ingest_llm_run_guard_max_reported_cost_usd=env.optional_float(
                "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_MAX_REPORTED_COST_USD"
            ),
            bulk_ingest_llm_run_guard_max_wall_time_seconds=(
                env.optional_float_default(
                    "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_MAX_WALL_TIME_SECONDS",
                    14400.0,
                )
            ),
            google_api_key=(
                env.get("ATAGIA_GOOGLE_API_KEY")
                or env.get("GEMINI_API_KEY")
                or env.get("GEMINI_KEY")
                or env.get("GOOGLE_API_KEY")
                or None
            ),
            allow_insecure_http=env.bool("ATAGIA_ALLOW_INSECURE_HTTP", False),
            default_language_code=env.language_code("ATAGIA_DEFAULT_LANGUAGE_CODE"),
            consequence_detector_card_concurrency=int(
                env.get("ATAGIA_CONSEQUENCE_DETECTOR_CARD_CONCURRENCY", "2")
            ),
            compactor_summary_card_concurrency=int(
                env.get("ATAGIA_COMPACTOR_SUMMARY_CARD_CONCURRENCY", "4")
            ),
            applicability_scorer_card_concurrency=int(
                env.get("ATAGIA_APPLICABILITY_SCORER_CARD_CONCURRENCY", "4")
            ),
            embedding_backend=env.get("ATAGIA_EMBEDDING_BACKEND", "none")
            .strip()
            .lower(),
            embedding_model=env.get("ATAGIA_EMBEDDING_MODEL")
            or DEFAULT_EMBEDDING_MODEL,
            embedding_dimension=int(env.get("ATAGIA_EMBEDDING_DIMENSION", "1536")),
            embedding_vector_limit_cap=int(
                env.get("ATAGIA_EMBEDDING_VECTOR_LIMIT_CAP", "50")
            ),
            embedding_search_overfetch_multiplier=int(
                env.get("ATAGIA_EMBEDDING_SEARCH_OVERFETCH_MULTIPLIER", "4")
            ),
            rrf_k=int(env.get("ATAGIA_RRF_K", "60")),
            recall_recovery_scoring_max_candidates=env.optional_int(
                "ATAGIA_RECALL_RECOVERY_SCORING_MAX_CANDIDATES"
            ),
            broad_list_comparable_rrf_enabled=env.bool(
                "ATAGIA_BROAD_LIST_COMPARABLE_RRF_ENABLED",
                False,
            ),
            fused_candidate_guard_ordering_enabled=env.bool(
                "ATAGIA_FUSED_CANDIDATE_GUARD_ORDERING_ENABLED",
                False,
            ),
            memory_fts_canonical_bm25_weight=float(
                env.get("ATAGIA_MEMORY_FTS_CANONICAL_BM25_WEIGHT", "1.2")
            ),
            memory_fts_index_bm25_weight=float(
                env.get("ATAGIA_MEMORY_FTS_INDEX_BM25_WEIGHT", "0.8")
            ),
            lifecycle_decay_days=int(env.get("ATAGIA_LIFECYCLE_DECAY_DAYS", "7")),
            lifecycle_decay_rate=float(env.get("ATAGIA_LIFECYCLE_DECAY_RATE", "0.9")),
            lifecycle_archive_vitality=float(
                env.get("ATAGIA_LIFECYCLE_ARCHIVE_VITALITY", "0.05")
            ),
            lifecycle_archive_confidence=float(
                env.get("ATAGIA_LIFECYCLE_ARCHIVE_CONFIDENCE", "0.3")
            ),
            ephemeral_scoring_hours=int(
                env.get("ATAGIA_EPHEMERAL_SCORING_HOURS", "24")
            ),
            lifecycle_ephemeral_ttl_hours=int(
                env.get("ATAGIA_LIFECYCLE_EPHEMERAL_TTL_HOURS", "24")
            ),
            lifecycle_review_ttl_days=int(
                env.get("ATAGIA_LIFECYCLE_REVIEW_TTL_DAYS", "7")
            ),
            promotion_conv_to_ws_min_conversations=int(
                env.get("ATAGIA_PROMOTION_CONV_TO_WS_MIN_CONVERSATIONS", "2")
            ),
            promotion_ws_to_global_min_sessions=int(
                env.get("ATAGIA_PROMOTION_WS_TO_GLOBAL_MIN_SESSIONS", "3")
            ),
            promotion_require_mode_consistency=env.bool(
                "ATAGIA_PROMOTION_REQUIRE_MODE_CONSISTENCY",
                True,
            ),
            belief_tension_increment=float(
                env.get("ATAGIA_BELIEF_TENSION_INCREMENT", "0.15")
            ),
            belief_tension_decrement=float(
                env.get("ATAGIA_BELIEF_TENSION_DECREMENT", "0.05")
            ),
            belief_tension_threshold=float(
                env.get("ATAGIA_BELIEF_TENSION_THRESHOLD", "0.5")
            ),
            skip_belief_revision=env.bool("ATAGIA_SKIP_BELIEF_REVISION", False),
            skip_compaction=env.bool("ATAGIA_SKIP_COMPACTION", False),
            episode_synthesis_max_episodes=int(
                env.get("ATAGIA_EPISODE_SYNTHESIS_MAX_EPISODES", "24")
            ),
            opf_privacy_filter_enabled=env.bool(
                "ATAGIA_OPF_PRIVACY_FILTER_ENABLED", False
            ),
            opf_primary_url=env.get(
                "ATAGIA_OPF_PRIMARY_URL", "http://127.0.0.1:8008"
            ),
            opf_fallback_url=env.get(
                "ATAGIA_OPF_FALLBACK_URL", "http://127.0.0.1:8008"
            ),
            opf_timeout_seconds=float(env.get("ATAGIA_OPF_TIMEOUT_SECONDS", "2.0")),
            privacy_validation_gate_enabled=env.bool(
                "ATAGIA_PRIVACY_VALIDATION_GATE_ENABLED",
                False,
            ),
            privacy_validation_gate_timeout_seconds=float(
                env.get("ATAGIA_PRIVACY_VALIDATION_GATE_TIMEOUT_SECONDS", "20.0")
            ),
            privacy_validation_gate_max_source_chars=int(
                env.get("ATAGIA_PRIVACY_VALIDATION_GATE_MAX_SOURCE_CHARS", "6000")
            ),
            privacy_validation_gate_max_summaries_gated_per_job=int(
                env.get(
                    "ATAGIA_PRIVACY_VALIDATION_GATE_MAX_SUMMARIES_GATED_PER_JOB", "50"
                )
            ),
            operational_high_risk_enabled=env.bool(
                "ATAGIA_OPERATIONAL_HIGH_RISK_ENABLED",
                False,
            ),
            operational_allowed_profiles=env.csv_tuple(
                "ATAGIA_OPERATIONAL_ALLOWED_PROFILES",
                ("normal", "low_power", "offline"),
            ),
            context_cache_enabled=env.bool("ATAGIA_CONTEXT_CACHE_ENABLED", True),
            initial_context_package_read_enabled=env.bool(
                "ATAGIA_INITIAL_CONTEXT_PACKAGE_READ_ENABLED",
                True,
            ),
            initial_context_package_refresh_enabled=env.bool(
                "ATAGIA_INITIAL_CONTEXT_PACKAGE_REFRESH_ENABLED",
                True,
            ),
            initial_context_package_curation_enabled=env.bool(
                "ATAGIA_INITIAL_CONTEXT_PACKAGE_CURATION_ENABLED",
                False,
            ),
            initial_context_package_prompt_max_tokens=int(
                env.get("ATAGIA_INITIAL_CONTEXT_PACKAGE_PROMPT_MAX_TOKENS", "900")
            ),
            initial_context_package_profile_max_tokens=int(
                env.get("ATAGIA_INITIAL_CONTEXT_PACKAGE_PROFILE_MAX_TOKENS", "700")
            ),
            initial_context_package_total_max_tokens=int(
                env.get("ATAGIA_INITIAL_CONTEXT_PACKAGE_TOTAL_MAX_TOKENS", "2200")
            ),
            initial_context_package_curated_block_max_tokens=int(
                env.get("ATAGIA_INITIAL_CONTEXT_PACKAGE_CURATED_BLOCK_MAX_TOKENS", "450")
            ),
            initial_context_package_curated_max_items=int(
                env.get("ATAGIA_INITIAL_CONTEXT_PACKAGE_CURATED_MAX_ITEMS", "8")
            ),
            initial_context_package_curation_max_output_tokens=int(
                env.get("ATAGIA_INITIAL_CONTEXT_PACKAGE_CURATION_MAX_OUTPUT_TOKENS", "2048")
            ),
            context_cache_min_ttl_seconds=int(
                env.get("ATAGIA_CONTEXT_CACHE_MIN_TTL_SECONDS", "60")
            ),
            context_cache_max_ttl_seconds=int(
                env.get("ATAGIA_CONTEXT_CACHE_MAX_TTL_SECONDS", "3600")
            ),
            temporary_default_ttl_seconds=env.optional_int(
                "ATAGIA_TEMPORARY_DEFAULT_TTL_SECONDS"
            ),
            temporary_default_purge_on_close=env.bool(
                "ATAGIA_TEMPORARY_DEFAULT_PURGE_ON_CLOSE",
                True,
            ),
            tombstone_retention_days=int(
                env.get("ATAGIA_TOMBSTONE_RETENTION_DAYS", "1825")
            ),
            erasure_purge_streams=env.bool("ATAGIA_ERASURE_PURGE_STREAMS", True),
            disable_chunking_extraction=env.bool(
                "ATAGIA_DISABLE_CHUNKING_EXTRACTION",
                False,
            ),
            chunking_extraction_threshold_tokens=int(
                env.get("ATAGIA_CHUNKING_EXTRACTION_THRESHOLD_TOKENS", "2048")
            ),
            extraction_watchdog_enabled=env.bool(
                "ATAGIA_EXTRACTION_WATCHDOG_ENABLED", True
            ),
            extraction_watchdog_allow_different_provider=env.bool(
                "ATAGIA_EXTRACTION_WATCHDOG_ALLOW_DIFFERENT_PROVIDER",
                False,
            ),
            extraction_watchdog_bounded_retry_max_items=int(
                env.get("ATAGIA_EXTRACTION_WATCHDOG_BOUNDED_RETRY_MAX_ITEMS", "8")
            ),
            extraction_watchdog_bounded_retry_max_output_tokens=int(
                env.get(
                    "ATAGIA_EXTRACTION_WATCHDOG_BOUNDED_RETRY_MAX_OUTPUT_TOKENS", "8192"
                )
            ),
            lifecycle_lazy_enabled=env.bool("ATAGIA_LIFECYCLE_LAZY_ENABLED", True),
            lifecycle_min_interval_seconds=int(
                env.get("ATAGIA_LIFECYCLE_MIN_INTERVAL_SECONDS", "3600")
            ),
            lifecycle_busy_timeout_ms=int(
                env.get("ATAGIA_LIFECYCLE_BUSY_TIMEOUT_MS", "1000")
            ),
            lifecycle_busy_backoff_seconds=int(
                env.get("ATAGIA_LIFECYCLE_BUSY_BACKOFF_SECONDS", "60")
            ),
            lifecycle_failure_backoff_seconds=int(
                env.get("ATAGIA_LIFECYCLE_FAILURE_BACKOFF_SECONDS", "300")
            ),
            lifecycle_worker_enabled=env.bool(
                "ATAGIA_LIFECYCLE_WORKER_ENABLED", False
            ),
            lifecycle_worker_interval_seconds=int(
                env.get("ATAGIA_LIFECYCLE_WORKER_INTERVAL_SECONDS", "3600")
            ),
            retrieval_packets_dry_run_enabled=env.bool(
                "ATAGIA_RETRIEVAL_PACKETS_DRY_RUN_ENABLED",
                False,
            ),
            retrieval_packets_write_enabled=env.bool(
                "ATAGIA_RETRIEVAL_PACKETS_WRITE_ENABLED",
                False,
            ),
            fact_facet_surfaces_enabled=env.bool(
                "ATAGIA_FACT_FACET_SURFACES_ENABLED",
                False,
            ),
            fact_facet_retrieval_enabled=env.bool(
                "ATAGIA_FACT_FACET_RETRIEVAL_ENABLED",
                False,
            ),
            fact_facet_structured_only=env.bool(
                "ATAGIA_FACT_FACET_STRUCTURED_ONLY",
                True,
            ),
            fact_facet_span_coadmission_enabled=env.bool(
                "ATAGIA_FACT_FACET_SPAN_COADMISSION_ENABLED",
                True,
            ),
            fact_facet_retrieval_limit=int(
                env.get("ATAGIA_FACT_FACET_RETRIEVAL_LIMIT", "12")
            ),
            fact_facet_retrieval_rrf_weight=float(
                env.get("ATAGIA_FACT_FACET_RETRIEVAL_RRF_WEIGHT", "1.1")
            ),
            applicability_gate_mode=env.get(
                "ATAGIA_APPLICABILITY_GATE_MODE",
                "off",
            ).strip().lower(),
            small_corpus_token_threshold_ratio=float(
                env.get("ATAGIA_SMALL_CORPUS_TOKEN_THRESHOLD_RATIO", "0.7")
            ),
            assistant_guidance_enabled=env.bool(
                "ATAGIA_ASSISTANT_GUIDANCE_ENABLED",
                True,
            ),
            response_mode=env.get("ATAGIA_RESPONSE_MODE", "normal")
            .strip()
            .lower(),
            adaptive_retrieval=env.bool("ATAGIA_ADAPTIVE_RETRIEVAL", True),
            answer_stance=env.get("ATAGIA_ANSWER_STANCE", "reactive")
            .strip()
            .lower(),
            answer_stance_prompt_variant=env.get(
                "ATAGIA_ANSWER_STANCE_PROMPT_VARIANT",
                "baseline",
            )
            .strip()
            .lower(),
            answer_postcondition_guard_enabled=env.bool(
                "ATAGIA_ANSWER_POSTCONDITION_GUARD_ENABLED",
                False,
            ),
            answer_postcondition_retry_max_output_tokens=int(
                env.get("ATAGIA_ANSWER_POSTCONDITION_RETRY_MAX_OUTPUT_TOKENS", "8192")
            ),
            context_envelope_budget_tokens=int(
                env.get("ATAGIA_CONTEXT_ENVELOPE_BUDGET_TOKENS", "8192")
            ),
            context_envelope_ratios=env.ratio_mapping(
                "ATAGIA_CONTEXT_ENVELOPE_RATIOS",
                CONTEXT_ENVELOPE_DEFAULT_RATIOS,
            ),
            benchmark_disable_raw_recent_transcript=env.bool(
                "ATAGIA_BENCHMARK_DISABLE_RAW_RECENT_TRANSCRIPT",
                False,
            ),
            recent_transcript_overage_ratio=float(
                env.get("ATAGIA_RECENT_TRANSCRIPT_OVERAGE_RATIO", "0.025")
            ),
            topic_working_set_enabled=env.bool(
                "ATAGIA_TOPIC_WORKING_SET_ENABLED", True
            ),
            topic_working_set_update_mode=env.get(
                "ATAGIA_TOPIC_WORKING_SET_UPDATE_MODE", "selective"
            ),
            topic_working_set_refresh_message_lag=int(
                env.get("ATAGIA_TOPIC_WORKING_SET_REFRESH_MESSAGE_LAG", "4")
            ),
            topic_working_set_stale_message_lag=int(
                env.get("ATAGIA_TOPIC_WORKING_SET_STALE_MESSAGE_LAG", "10")
            ),
            topic_working_set_refresh_token_lag=int(
                env.get("ATAGIA_TOPIC_WORKING_SET_REFRESH_TOKEN_LAG", "2000")
            ),
            topic_working_set_stale_token_lag=int(
                env.get("ATAGIA_TOPIC_WORKING_SET_STALE_TOKEN_LAG", "5000")
            ),
            topic_working_set_refresh_batch_messages=int(
                env.get("ATAGIA_TOPIC_WORKING_SET_REFRESH_BATCH_MESSAGES", "8")
            ),
            graph_projection_enabled=env.bool(
                "ATAGIA_GRAPH_PROJECTION_ENABLED", False
            ),
            verbatim_evidence_search_enabled=env.bool(
                "ATAGIA_VERBATIM_EVIDENCE_SEARCH_ENABLED",
                True,
            ),
            verbatim_evidence_search_rrf_weight=float(
                env.get("ATAGIA_VERBATIM_EVIDENCE_SEARCH_RRF_WEIGHT", "0.75")
            ),
            verbatim_evidence_search_limit=int(
                env.get("ATAGIA_VERBATIM_EVIDENCE_SEARCH_LIMIT", "8")
            ),
            verbatim_evidence_window_size=int(
                env.get("ATAGIA_VERBATIM_EVIDENCE_WINDOW_SIZE", "3")
            ),
            verbatim_evidence_window_overlap=int(
                env.get("ATAGIA_VERBATIM_EVIDENCE_WINDOW_OVERLAP", "1")
            ),
            openai_proxy_model_id=env.get(
                "ATAGIA_PROXY_MODEL_ID",
                "atagia-memory-proxy",
            ),
            openai_proxy_upstream_model=env.optional_str(
                "ATAGIA_PROXY_UPSTREAM_MODEL"
            ),
            openai_proxy_default_mode=env.optional_str("ATAGIA_PROXY_DEFAULT_MODE"),
            openai_proxy_max_output_tokens=int(
                env.get("ATAGIA_PROXY_MAX_OUTPUT_TOKENS", "8192")
            ),
            request_max_body_bytes=int(
                env.get("ATAGIA_REQUEST_MAX_BODY_BYTES", str(32 * 1024 * 1024))
            ),
            request_max_message_text_bytes=int(
                env.get("ATAGIA_REQUEST_MAX_MESSAGE_TEXT_BYTES", str(256 * 1024))
            ),
            request_max_attachments=int(
                env.get("ATAGIA_REQUEST_MAX_ATTACHMENTS", "16")
            ),
            request_max_attachment_decoded_bytes=int(
                env.get(
                    "ATAGIA_REQUEST_MAX_ATTACHMENT_DECODED_BYTES",
                    str(10 * 1024 * 1024),
                )
            ),
            request_max_attachments_decoded_bytes=int(
                env.get(
                    "ATAGIA_REQUEST_MAX_ATTACHMENTS_DECODED_BYTES",
                    str(20 * 1024 * 1024),
                )
            ),
            request_max_metadata_bytes=int(
                env.get("ATAGIA_REQUEST_MAX_METADATA_BYTES", str(64 * 1024))
            ),
            cors_allowed_origins=env.csv_tuple(
                "ATAGIA_CORS_ALLOWED_ORIGINS",
                (),
            ),
        )

    def migrations_dir(self, root: Path | None = None) -> Path:
        base = root or Path.cwd()
        return (base / self.migrations_path).resolve()

    def manifests_dir(self, root: Path | None = None) -> Path:
        base = root or Path.cwd()
        return (base / self.manifests_path).resolve()

    def operational_profiles_dir(self, root: Path | None = None) -> Path:
        base = root or Path.cwd()
        return (base / self.operational_profiles_path).resolve()

    def artifact_blobs_dir(self, root: Path | None = None) -> Path:
        base = root or Path.cwd()
        return (base / self.artifact_blob_storage_path).resolve()
