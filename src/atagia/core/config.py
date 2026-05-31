"""Configuration loading from environment variables."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path

from dotenv import load_dotenv

from atagia.core.env import env_bool as _env_bool
from atagia.memory.context_envelope import (
    CONTEXT_ENVELOPE_DEFAULT_RATIOS,
    allocate_context_envelope_budget,
)
from atagia.services.model_resolution import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_STRUCTURED_OUTPUT_RESCUE_MODEL,
    component_env_models_from_env,
    intimacy_component_env_models_from_env,
)


_DOTENV_LOADED = False
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


def _repo_resource_path(name: str) -> Path | None:
    repo_root = _repo_root()
    if repo_root is None:
        return None
    candidate = repo_root / name
    if candidate.exists():
        return candidate
    return None


def default_resource_path(name: str) -> str:
    """Return a repo resource path, falling back to packaged resources."""
    repo_path = _repo_resource_path(name)
    if repo_path is not None:
        return str(repo_path)
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

    if path == Path(name) or path.as_posix().lstrip("./") == name:
        return default_resource_path(name)

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


def _env_csv_tuple(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = tuple(part.strip() for part in value.split(",") if part.strip())
    return normalized or default


def _env_optional_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return int(value)


def _env_optional_int_default(name: str, default: int) -> int | None:
    return _env_optional_int(name) if os.getenv(name) is not None else default


def _env_optional_float(name: str) -> float | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return float(value)


def _env_optional_float_default(name: str, default: float) -> float | None:
    return _env_optional_float(name) if os.getenv(name) is not None else default


def _env_optional_str(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _env_ratio_mapping(name: str, default: dict[str, float]) -> dict[str, float]:
    value = os.getenv(name)
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


MIN_RECENT_TRANSCRIPT_BUDGET_TOKENS = 2048


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
    llm_run_guard_enabled: bool = True
    llm_run_guard_mode: str = "enforce"
    llm_run_guard_max_total_calls: int | None = None
    llm_run_guard_max_total_failed_calls: int | None = 80
    llm_run_guard_max_failed_call_ratio: float | None = 0.50
    llm_run_guard_failed_ratio_min_calls: int = 20
    llm_run_guard_max_failed_calls_per_purpose: int | None = None
    llm_run_guard_max_failed_ratio_per_purpose: float | None = 0.50
    llm_run_guard_purpose_failure_ratio_min_calls: int = 10
    llm_run_guard_max_consecutive_failures_per_purpose: int | None = 8
    llm_run_guard_max_total_tokens: int | None = None
    llm_run_guard_max_reported_cost_usd: float | None = None
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
    anthropic_base_url: str | None = None
    anthropic_request_timeout_seconds: float = 120.0
    openai_base_url: str | None = None
    openai_embedding_base_url: str | None = None
    openrouter_base_url: str | None = None
    llm_forced_global_model: str | None = None
    llm_ingest_model: str | None = None
    llm_retrieval_model: str | None = None
    llm_component_models: dict[str, str] = field(default_factory=dict)
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
    llm_debug_io_dir: str = "./docs/tmp/llm_debug"
    llm_debug_io_purposes: tuple[str, ...] = ()
    llm_debug_io_raw: bool = False
    llm_debug_io_max_chars: int = 50_000
    operational_profiles_path: str = "./operational_profiles"
    artifact_blob_storage_kind: str = "sqlite_blob"
    artifact_blob_storage_path: str = "./data/artifact_blobs"
    allow_admin_export_anonymization: bool = False
    allow_insecure_http: bool = False
    embedding_backend: str = "none"
    embedding_model: str | None = None
    embedding_dimension: int = 1536
    embedding_vector_limit_cap: int = 50
    embedding_search_overfetch_multiplier: int = 4
    rrf_k: int = 60
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
    answer_stance: str = "reactive"
    answer_stance_prompt_variant: str = "baseline"
    answer_postcondition_guard_enabled: bool = False
    answer_postcondition_retry_max_output_tokens: int = 8192
    recent_transcript_budget_tokens: int | None = None
    context_envelope_budget_tokens: int = 4096
    context_envelope_ratios: dict[str, float] = field(
        default_factory=lambda: dict(CONTEXT_ENVELOPE_DEFAULT_RATIOS)
    )
    benchmark_disable_raw_recent_transcript: bool = False
    recent_transcript_overage_ratio: float = 0.025
    topic_working_set_enabled: bool = True
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
        if not self.openai_proxy_model_id.strip():
            raise ValueError("openai_proxy_model_id cannot be blank")
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
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
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
        if (
            self.temporary_default_ttl_seconds is not None
            and self.temporary_default_ttl_seconds <= 0
        ):
            raise ValueError("temporary_default_ttl_seconds must be positive when set")
        if self.tombstone_retention_days <= 0:
            raise ValueError("tombstone_retention_days must be positive")
        if self.anthropic_request_timeout_seconds <= 0:
            raise ValueError("anthropic_request_timeout_seconds must be positive")
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
                "artifact_blob_storage_kind must be 'sqlite_blob' or 'local_file'"
            )
        if (
            self.artifact_blob_storage_kind == "local_file"
            and not self.artifact_blob_storage_path.strip()
        ):
            raise ValueError(
                "artifact_blob_storage_path is required for local_file artifact storage"
            )
        if not 0.0 <= self.small_corpus_token_threshold_ratio <= 1.0:
            raise ValueError(
                "small_corpus_token_threshold_ratio must be in the interval [0.0, 1.0]"
            )
        if (
            self.recent_transcript_budget_tokens is not None
            and self.recent_transcript_budget_tokens <= 0
        ):
            raise ValueError("recent_transcript_budget_tokens must be positive")
        allocate_context_envelope_budget(
            self.context_envelope_budget_tokens,
            self.context_envelope_ratios,
        )
        if not 0.0 <= self.recent_transcript_overage_ratio <= 1.0:
            raise ValueError(
                "recent_transcript_overage_ratio must be in the interval [0.0, 1.0]"
            )
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
    def from_env(cls) -> "Settings":
        _load_dotenv_once()
        return cls(
            sqlite_path=os.getenv("ATAGIA_SQLITE_PATH", "./data/atagia.db"),
            migrations_path=configured_resource_path(
                "migrations",
                os.getenv("ATAGIA_MIGRATIONS_PATH"),
            ),
            manifests_path=configured_resource_path(
                "manifests",
                os.getenv("ATAGIA_MANIFESTS_PATH"),
            ),
            operational_profiles_path=configured_resource_path(
                "operational_profiles",
                os.getenv("ATAGIA_OPERATIONAL_PROFILES_PATH"),
            ),
            artifact_blob_storage_kind=os.getenv(
                "ATAGIA_ARTIFACT_BLOB_STORAGE_KIND",
                "sqlite_blob",
            )
            .strip()
            .lower(),
            artifact_blob_storage_path=os.getenv(
                "ATAGIA_ARTIFACT_BLOB_STORAGE_PATH",
                "./data/artifact_blobs",
            ),
            storage_backend=os.getenv("ATAGIA_STORAGE_BACKEND", "inprocess")
            .strip()
            .lower(),
            redis_url=os.getenv("ATAGIA_REDIS_URL", "redis://localhost:6379/0"),
            anthropic_api_key=os.getenv("ATAGIA_ANTHROPIC_API_KEY") or None,
            openai_api_key=os.getenv("ATAGIA_OPENAI_API_KEY") or None,
            openrouter_api_key=os.getenv("ATAGIA_OPENROUTER_API_KEY") or None,
            anthropic_base_url=os.getenv("ATAGIA_ANTHROPIC_BASE_URL") or None,
            anthropic_request_timeout_seconds=float(
                os.getenv("ATAGIA_ANTHROPIC_REQUEST_TIMEOUT_SECONDS", "120.0")
            ),
            openai_base_url=os.getenv("ATAGIA_OPENAI_BASE_URL") or None,
            openai_embedding_base_url=(
                os.getenv("ATAGIA_OPENAI_EMBEDDING_BASE_URL") or None
            ),
            openrouter_base_url=os.getenv("ATAGIA_OPENROUTER_BASE_URL") or None,
            openrouter_site_url=os.getenv(
                "ATAGIA_OPENROUTER_SITE_URL", "http://localhost"
            ),
            openrouter_app_name=os.getenv("ATAGIA_OPENROUTER_APP_NAME", "Atagia"),
            llm_chat_model=os.getenv("ATAGIA_LLM_CHAT_MODEL") or None,
            llm_forced_global_model=os.getenv("ATAGIA_LLM_FORCED_GLOBAL_MODEL") or None,
            llm_ingest_model=os.getenv("ATAGIA_LLM_INGEST_MODEL") or None,
            llm_retrieval_model=os.getenv("ATAGIA_LLM_RETRIEVAL_MODEL") or None,
            llm_component_models=component_env_models_from_env(os.environ),
            llm_intimacy_ingest_model=os.getenv("ATAGIA_LLM_INTIMACY_INGEST_MODEL")
            or None,
            llm_intimacy_retrieval_model=(
                os.getenv("ATAGIA_LLM_INTIMACY_RETRIEVAL_MODEL") or None
            ),
            llm_intimacy_component_models=intimacy_component_env_models_from_env(
                os.environ
            ),
            llm_intimacy_proactive_routing_enabled=_env_bool(
                "ATAGIA_LLM_INTIMACY_PROACTIVE_ROUTING_ENABLED",
                False,
            ),
            llm_structured_output_retry_attempts=int(
                os.getenv("ATAGIA_LLM_STRUCTURED_OUTPUT_RETRY_ATTEMPTS", "1")
            ),
            llm_structured_output_rescue_enabled=_env_bool(
                "ATAGIA_LLM_STRUCTURED_OUTPUT_RESCUE_ENABLED",
                False,
            ),
            llm_structured_output_rescue_model=(
                _env_optional_str("ATAGIA_LLM_STRUCTURED_OUTPUT_RESCUE_MODEL")
                or DEFAULT_STRUCTURED_OUTPUT_RESCUE_MODEL
            ),
            llm_technical_recovery_enabled=_env_bool(
                "ATAGIA_LLM_TECHNICAL_RECOVERY_ENABLED",
                True,
            ),
            llm_output_limit_retry_attempts=int(
                os.getenv("ATAGIA_LLM_OUTPUT_LIMIT_RETRY_ATTEMPTS", "1")
            ),
            llm_runaway_watchdog_enabled=_env_bool(
                "ATAGIA_LLM_RUNAWAY_WATCHDOG_ENABLED",
                True,
            ),
            llm_runaway_min_elapsed_seconds=float(
                os.getenv("ATAGIA_LLM_RUNAWAY_MIN_ELAPSED_SECONDS", "8.0")
            ),
            llm_runaway_min_output_tokens=int(
                os.getenv("ATAGIA_LLM_RUNAWAY_MIN_OUTPUT_TOKENS", "2048")
            ),
            llm_runaway_check_interval_tokens=int(
                os.getenv("ATAGIA_LLM_RUNAWAY_CHECK_INTERVAL_TOKENS", "1024")
            ),
            llm_runaway_max_checks=int(
                os.getenv("ATAGIA_LLM_RUNAWAY_MAX_CHECKS", "2")
            ),
            llm_runaway_hard_abort_min_output_tokens=int(
                os.getenv("ATAGIA_LLM_RUNAWAY_HARD_ABORT_MIN_OUTPUT_TOKENS", "4096")
            ),
            llm_runaway_min_repeat_count=int(
                os.getenv("ATAGIA_LLM_RUNAWAY_MIN_REPEAT_COUNT", "3")
            ),
            llm_runaway_min_repeat_ratio_tokens=float(
                os.getenv("ATAGIA_LLM_RUNAWAY_MIN_REPEAT_RATIO_TOKENS", "0.12")
            ),
            llm_runaway_output_input_ratio=float(
                os.getenv("ATAGIA_LLM_RUNAWAY_OUTPUT_INPUT_RATIO", "12.0")
            ),
            llm_runaway_hard_output_input_ratio=float(
                os.getenv("ATAGIA_LLM_RUNAWAY_HARD_OUTPUT_INPUT_RATIO", "8.0")
            ),
            llm_debug_io_enabled=_env_bool("ATAGIA_DEBUG_LLM_IO", False),
            llm_debug_io_dir=os.getenv(
                "ATAGIA_DEBUG_LLM_IO_DIR",
                "./docs/tmp/llm_debug",
            ),
            llm_debug_io_purposes=_env_csv_tuple("ATAGIA_DEBUG_LLM_IO_PURPOSES", ()),
            llm_debug_io_raw=_env_bool("ATAGIA_DEBUG_LLM_IO_RAW", False),
            llm_debug_io_max_chars=int(
                os.getenv("ATAGIA_DEBUG_LLM_IO_MAX_CHARS", "50000")
            ),
            service_mode=_env_bool("ATAGIA_SERVICE_MODE", False),
            service_api_key=os.getenv("ATAGIA_SERVICE_API_KEY") or None,
            admin_api_key=os.getenv("ATAGIA_ADMIN_API_KEY") or None,
            allow_admin_export_anonymization=_env_bool(
                "ATAGIA_ALLOW_ADMIN_EXPORT_ANONYMIZATION",
                False,
            ),
            workers_enabled=_env_bool("ATAGIA_WORKERS_ENABLED", False),
            debug=_env_bool("ATAGIA_DEBUG", False),
            worker_circuit_breaker_enabled=_env_bool(
                "ATAGIA_WORKER_CIRCUIT_BREAKER_ENABLED",
                True,
            ),
            worker_circuit_breaker_failure_threshold=int(
                os.getenv("ATAGIA_WORKER_CIRCUIT_BREAKER_FAILURE_THRESHOLD", "20")
            ),
            worker_circuit_breaker_window_seconds=int(
                os.getenv("ATAGIA_WORKER_CIRCUIT_BREAKER_WINDOW_SECONDS", "180")
            ),
            worker_circuit_breaker_min_failure_ratio=float(
                os.getenv("ATAGIA_WORKER_CIRCUIT_BREAKER_MIN_FAILURE_RATIO", "0.8")
            ),
            llm_run_guard_enabled=_env_bool("ATAGIA_LLM_RUN_GUARD_ENABLED", True),
            llm_run_guard_mode=os.getenv("ATAGIA_LLM_RUN_GUARD_MODE", "enforce")
            .strip()
            .lower(),
            llm_run_guard_max_total_calls=_env_optional_int(
                "ATAGIA_LLM_RUN_GUARD_MAX_TOTAL_CALLS"
            ),
            llm_run_guard_max_total_failed_calls=_env_optional_int_default(
                "ATAGIA_LLM_RUN_GUARD_MAX_TOTAL_FAILED_CALLS",
                80,
            ),
            llm_run_guard_max_failed_call_ratio=_env_optional_float_default(
                "ATAGIA_LLM_RUN_GUARD_MAX_FAILED_CALL_RATIO",
                0.50,
            ),
            llm_run_guard_failed_ratio_min_calls=int(
                os.getenv("ATAGIA_LLM_RUN_GUARD_FAILED_RATIO_MIN_CALLS", "20")
            ),
            llm_run_guard_max_failed_calls_per_purpose=_env_optional_int(
                "ATAGIA_LLM_RUN_GUARD_MAX_FAILED_CALLS_PER_PURPOSE"
            ),
            llm_run_guard_max_failed_ratio_per_purpose=_env_optional_float_default(
                "ATAGIA_LLM_RUN_GUARD_MAX_FAILED_RATIO_PER_PURPOSE",
                0.50,
            ),
            llm_run_guard_purpose_failure_ratio_min_calls=int(
                os.getenv(
                    "ATAGIA_LLM_RUN_GUARD_PURPOSE_FAILURE_RATIO_MIN_CALLS",
                    "10",
                )
            ),
            llm_run_guard_max_consecutive_failures_per_purpose=(
                _env_optional_int_default(
                    "ATAGIA_LLM_RUN_GUARD_MAX_CONSECUTIVE_FAILURES_PER_PURPOSE",
                    8,
                )
            ),
            llm_run_guard_max_total_tokens=_env_optional_int(
                "ATAGIA_LLM_RUN_GUARD_MAX_TOTAL_TOKENS"
            ),
            llm_run_guard_max_reported_cost_usd=_env_optional_float(
                "ATAGIA_LLM_RUN_GUARD_MAX_REPORTED_COST_USD"
            ),
            bulk_ingest_llm_run_guard_enabled=_env_bool(
                "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_ENABLED",
                True,
            ),
            bulk_ingest_llm_run_guard_max_total_calls=_env_optional_int_default(
                "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_MAX_TOTAL_CALLS",
                10000,
            ),
            bulk_ingest_llm_run_guard_max_total_failed_calls=(
                _env_optional_int_default(
                    "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_MAX_TOTAL_FAILED_CALLS",
                    40,
                )
            ),
            bulk_ingest_llm_run_guard_max_failed_call_ratio=(
                _env_optional_float_default(
                    "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_MAX_FAILED_CALL_RATIO",
                    0.20,
                )
            ),
            bulk_ingest_llm_run_guard_failed_ratio_min_calls=int(
                os.getenv(
                    "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_FAILED_RATIO_MIN_CALLS",
                    "20",
                )
            ),
            bulk_ingest_llm_run_guard_max_failed_calls_per_purpose=_env_optional_int(
                "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_MAX_FAILED_CALLS_PER_PURPOSE"
            ),
            bulk_ingest_llm_run_guard_max_failed_ratio_per_purpose=(
                _env_optional_float_default(
                    "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_MAX_FAILED_RATIO_PER_PURPOSE",
                    0.30,
                )
            ),
            bulk_ingest_llm_run_guard_purpose_failure_ratio_min_calls=int(
                os.getenv(
                    "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_PURPOSE_FAILURE_RATIO_MIN_CALLS",
                    "10",
                )
            ),
            bulk_ingest_llm_run_guard_max_consecutive_failures_per_purpose=(
                _env_optional_int_default(
                    "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_MAX_CONSECUTIVE_FAILURES_PER_PURPOSE",
                    4,
                )
            ),
            bulk_ingest_llm_run_guard_max_total_tokens=_env_optional_int(
                "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_MAX_TOTAL_TOKENS"
            ),
            bulk_ingest_llm_run_guard_max_reported_cost_usd=_env_optional_float(
                "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_MAX_REPORTED_COST_USD"
            ),
            bulk_ingest_llm_run_guard_max_wall_time_seconds=(
                _env_optional_float_default(
                    "ATAGIA_BULK_INGEST_LLM_RUN_GUARD_MAX_WALL_TIME_SECONDS",
                    14400.0,
                )
            ),
            google_api_key=(
                os.getenv("ATAGIA_GOOGLE_API_KEY")
                or os.getenv("GEMINI_API_KEY")
                or os.getenv("GEMINI_KEY")
                or os.getenv("GOOGLE_API_KEY")
                or None
            ),
            allow_insecure_http=_env_bool("ATAGIA_ALLOW_INSECURE_HTTP", False),
            embedding_backend=os.getenv("ATAGIA_EMBEDDING_BACKEND", "none")
            .strip()
            .lower(),
            embedding_model=os.getenv("ATAGIA_EMBEDDING_MODEL")
            or DEFAULT_EMBEDDING_MODEL,
            embedding_dimension=int(os.getenv("ATAGIA_EMBEDDING_DIMENSION", "1536")),
            embedding_vector_limit_cap=int(
                os.getenv("ATAGIA_EMBEDDING_VECTOR_LIMIT_CAP", "50")
            ),
            embedding_search_overfetch_multiplier=int(
                os.getenv("ATAGIA_EMBEDDING_SEARCH_OVERFETCH_MULTIPLIER", "4")
            ),
            rrf_k=int(os.getenv("ATAGIA_RRF_K", "60")),
            memory_fts_canonical_bm25_weight=float(
                os.getenv("ATAGIA_MEMORY_FTS_CANONICAL_BM25_WEIGHT", "1.2")
            ),
            memory_fts_index_bm25_weight=float(
                os.getenv("ATAGIA_MEMORY_FTS_INDEX_BM25_WEIGHT", "0.8")
            ),
            lifecycle_decay_days=int(os.getenv("ATAGIA_LIFECYCLE_DECAY_DAYS", "7")),
            lifecycle_decay_rate=float(os.getenv("ATAGIA_LIFECYCLE_DECAY_RATE", "0.9")),
            lifecycle_archive_vitality=float(
                os.getenv("ATAGIA_LIFECYCLE_ARCHIVE_VITALITY", "0.05")
            ),
            lifecycle_archive_confidence=float(
                os.getenv("ATAGIA_LIFECYCLE_ARCHIVE_CONFIDENCE", "0.3")
            ),
            ephemeral_scoring_hours=int(
                os.getenv("ATAGIA_EPHEMERAL_SCORING_HOURS", "24")
            ),
            lifecycle_ephemeral_ttl_hours=int(
                os.getenv("ATAGIA_LIFECYCLE_EPHEMERAL_TTL_HOURS", "24")
            ),
            lifecycle_review_ttl_days=int(
                os.getenv("ATAGIA_LIFECYCLE_REVIEW_TTL_DAYS", "7")
            ),
            promotion_conv_to_ws_min_conversations=int(
                os.getenv("ATAGIA_PROMOTION_CONV_TO_WS_MIN_CONVERSATIONS", "2")
            ),
            promotion_ws_to_global_min_sessions=int(
                os.getenv("ATAGIA_PROMOTION_WS_TO_GLOBAL_MIN_SESSIONS", "3")
            ),
            promotion_require_mode_consistency=_env_bool(
                "ATAGIA_PROMOTION_REQUIRE_MODE_CONSISTENCY",
                True,
            ),
            belief_tension_increment=float(
                os.getenv("ATAGIA_BELIEF_TENSION_INCREMENT", "0.15")
            ),
            belief_tension_decrement=float(
                os.getenv("ATAGIA_BELIEF_TENSION_DECREMENT", "0.05")
            ),
            belief_tension_threshold=float(
                os.getenv("ATAGIA_BELIEF_TENSION_THRESHOLD", "0.5")
            ),
            skip_belief_revision=_env_bool("ATAGIA_SKIP_BELIEF_REVISION", False),
            skip_compaction=_env_bool("ATAGIA_SKIP_COMPACTION", False),
            episode_synthesis_max_episodes=int(
                os.getenv("ATAGIA_EPISODE_SYNTHESIS_MAX_EPISODES", "24")
            ),
            opf_privacy_filter_enabled=_env_bool(
                "ATAGIA_OPF_PRIVACY_FILTER_ENABLED", False
            ),
            opf_primary_url=os.getenv(
                "ATAGIA_OPF_PRIMARY_URL", "http://127.0.0.1:8008"
            ),
            opf_fallback_url=os.getenv(
                "ATAGIA_OPF_FALLBACK_URL", "http://127.0.0.1:8008"
            ),
            opf_timeout_seconds=float(os.getenv("ATAGIA_OPF_TIMEOUT_SECONDS", "2.0")),
            privacy_validation_gate_enabled=_env_bool(
                "ATAGIA_PRIVACY_VALIDATION_GATE_ENABLED",
                False,
            ),
            privacy_validation_gate_timeout_seconds=float(
                os.getenv("ATAGIA_PRIVACY_VALIDATION_GATE_TIMEOUT_SECONDS", "20.0")
            ),
            privacy_validation_gate_max_source_chars=int(
                os.getenv("ATAGIA_PRIVACY_VALIDATION_GATE_MAX_SOURCE_CHARS", "6000")
            ),
            privacy_validation_gate_max_summaries_gated_per_job=int(
                os.getenv(
                    "ATAGIA_PRIVACY_VALIDATION_GATE_MAX_SUMMARIES_GATED_PER_JOB", "50"
                )
            ),
            operational_high_risk_enabled=_env_bool(
                "ATAGIA_OPERATIONAL_HIGH_RISK_ENABLED",
                False,
            ),
            operational_allowed_profiles=_env_csv_tuple(
                "ATAGIA_OPERATIONAL_ALLOWED_PROFILES",
                ("normal", "low_power", "offline"),
            ),
            context_cache_enabled=_env_bool("ATAGIA_CONTEXT_CACHE_ENABLED", True),
            initial_context_package_read_enabled=_env_bool(
                "ATAGIA_INITIAL_CONTEXT_PACKAGE_READ_ENABLED",
                True,
            ),
            initial_context_package_refresh_enabled=_env_bool(
                "ATAGIA_INITIAL_CONTEXT_PACKAGE_REFRESH_ENABLED",
                True,
            ),
            initial_context_package_curation_enabled=_env_bool(
                "ATAGIA_INITIAL_CONTEXT_PACKAGE_CURATION_ENABLED",
                False,
            ),
            initial_context_package_prompt_max_tokens=int(
                os.getenv("ATAGIA_INITIAL_CONTEXT_PACKAGE_PROMPT_MAX_TOKENS", "900")
            ),
            initial_context_package_profile_max_tokens=int(
                os.getenv("ATAGIA_INITIAL_CONTEXT_PACKAGE_PROFILE_MAX_TOKENS", "700")
            ),
            initial_context_package_total_max_tokens=int(
                os.getenv("ATAGIA_INITIAL_CONTEXT_PACKAGE_TOTAL_MAX_TOKENS", "2200")
            ),
            initial_context_package_curated_block_max_tokens=int(
                os.getenv("ATAGIA_INITIAL_CONTEXT_PACKAGE_CURATED_BLOCK_MAX_TOKENS", "450")
            ),
            initial_context_package_curated_max_items=int(
                os.getenv("ATAGIA_INITIAL_CONTEXT_PACKAGE_CURATED_MAX_ITEMS", "8")
            ),
            initial_context_package_curation_max_output_tokens=int(
                os.getenv("ATAGIA_INITIAL_CONTEXT_PACKAGE_CURATION_MAX_OUTPUT_TOKENS", "2048")
            ),
            context_cache_min_ttl_seconds=int(
                os.getenv("ATAGIA_CONTEXT_CACHE_MIN_TTL_SECONDS", "60")
            ),
            context_cache_max_ttl_seconds=int(
                os.getenv("ATAGIA_CONTEXT_CACHE_MAX_TTL_SECONDS", "3600")
            ),
            temporary_default_ttl_seconds=_env_optional_int(
                "ATAGIA_TEMPORARY_DEFAULT_TTL_SECONDS"
            ),
            temporary_default_purge_on_close=_env_bool(
                "ATAGIA_TEMPORARY_DEFAULT_PURGE_ON_CLOSE",
                True,
            ),
            tombstone_retention_days=int(
                os.getenv("ATAGIA_TOMBSTONE_RETENTION_DAYS", "1825")
            ),
            erasure_purge_streams=_env_bool("ATAGIA_ERASURE_PURGE_STREAMS", True),
            disable_chunking_extraction=_env_bool(
                "ATAGIA_DISABLE_CHUNKING_EXTRACTION",
                False,
            ),
            chunking_extraction_threshold_tokens=int(
                os.getenv("ATAGIA_CHUNKING_EXTRACTION_THRESHOLD_TOKENS", "2048")
            ),
            extraction_watchdog_enabled=_env_bool(
                "ATAGIA_EXTRACTION_WATCHDOG_ENABLED", True
            ),
            extraction_watchdog_allow_different_provider=_env_bool(
                "ATAGIA_EXTRACTION_WATCHDOG_ALLOW_DIFFERENT_PROVIDER",
                False,
            ),
            extraction_watchdog_bounded_retry_max_items=int(
                os.getenv("ATAGIA_EXTRACTION_WATCHDOG_BOUNDED_RETRY_MAX_ITEMS", "8")
            ),
            extraction_watchdog_bounded_retry_max_output_tokens=int(
                os.getenv(
                    "ATAGIA_EXTRACTION_WATCHDOG_BOUNDED_RETRY_MAX_OUTPUT_TOKENS", "8192"
                )
            ),
            lifecycle_lazy_enabled=_env_bool("ATAGIA_LIFECYCLE_LAZY_ENABLED", True),
            lifecycle_min_interval_seconds=int(
                os.getenv("ATAGIA_LIFECYCLE_MIN_INTERVAL_SECONDS", "3600")
            ),
            lifecycle_busy_timeout_ms=int(
                os.getenv("ATAGIA_LIFECYCLE_BUSY_TIMEOUT_MS", "1000")
            ),
            lifecycle_busy_backoff_seconds=int(
                os.getenv("ATAGIA_LIFECYCLE_BUSY_BACKOFF_SECONDS", "60")
            ),
            lifecycle_failure_backoff_seconds=int(
                os.getenv("ATAGIA_LIFECYCLE_FAILURE_BACKOFF_SECONDS", "300")
            ),
            lifecycle_worker_enabled=_env_bool(
                "ATAGIA_LIFECYCLE_WORKER_ENABLED", False
            ),
            lifecycle_worker_interval_seconds=int(
                os.getenv("ATAGIA_LIFECYCLE_WORKER_INTERVAL_SECONDS", "3600")
            ),
            retrieval_packets_dry_run_enabled=_env_bool(
                "ATAGIA_RETRIEVAL_PACKETS_DRY_RUN_ENABLED",
                False,
            ),
            retrieval_packets_write_enabled=_env_bool(
                "ATAGIA_RETRIEVAL_PACKETS_WRITE_ENABLED",
                False,
            ),
            fact_facet_surfaces_enabled=_env_bool(
                "ATAGIA_FACT_FACET_SURFACES_ENABLED",
                False,
            ),
            fact_facet_retrieval_enabled=_env_bool(
                "ATAGIA_FACT_FACET_RETRIEVAL_ENABLED",
                False,
            ),
            fact_facet_structured_only=_env_bool(
                "ATAGIA_FACT_FACET_STRUCTURED_ONLY",
                True,
            ),
            fact_facet_span_coadmission_enabled=_env_bool(
                "ATAGIA_FACT_FACET_SPAN_COADMISSION_ENABLED",
                True,
            ),
            fact_facet_retrieval_limit=int(
                os.getenv("ATAGIA_FACT_FACET_RETRIEVAL_LIMIT", "12")
            ),
            fact_facet_retrieval_rrf_weight=float(
                os.getenv("ATAGIA_FACT_FACET_RETRIEVAL_RRF_WEIGHT", "1.1")
            ),
            applicability_gate_mode=os.getenv(
                "ATAGIA_APPLICABILITY_GATE_MODE",
                "off",
            ).strip().lower(),
            small_corpus_token_threshold_ratio=float(
                os.getenv("ATAGIA_SMALL_CORPUS_TOKEN_THRESHOLD_RATIO", "0.7")
            ),
            assistant_guidance_enabled=_env_bool(
                "ATAGIA_ASSISTANT_GUIDANCE_ENABLED",
                True,
            ),
            response_mode=os.getenv("ATAGIA_RESPONSE_MODE", "normal")
            .strip()
            .lower(),
            answer_stance=os.getenv("ATAGIA_ANSWER_STANCE", "reactive")
            .strip()
            .lower(),
            answer_stance_prompt_variant=os.getenv(
                "ATAGIA_ANSWER_STANCE_PROMPT_VARIANT",
                "baseline",
            )
            .strip()
            .lower(),
            answer_postcondition_guard_enabled=_env_bool(
                "ATAGIA_ANSWER_POSTCONDITION_GUARD_ENABLED",
                False,
            ),
            answer_postcondition_retry_max_output_tokens=int(
                os.getenv("ATAGIA_ANSWER_POSTCONDITION_RETRY_MAX_OUTPUT_TOKENS", "8192")
            ),
            recent_transcript_budget_tokens=_env_optional_int(
                "ATAGIA_RECENT_TRANSCRIPT_BUDGET_TOKENS"
            ),
            context_envelope_budget_tokens=int(
                os.getenv("ATAGIA_CONTEXT_ENVELOPE_BUDGET_TOKENS", "4096")
            ),
            context_envelope_ratios=_env_ratio_mapping(
                "ATAGIA_CONTEXT_ENVELOPE_RATIOS",
                CONTEXT_ENVELOPE_DEFAULT_RATIOS,
            ),
            benchmark_disable_raw_recent_transcript=_env_bool(
                "ATAGIA_BENCHMARK_DISABLE_RAW_RECENT_TRANSCRIPT",
                False,
            ),
            recent_transcript_overage_ratio=float(
                os.getenv("ATAGIA_RECENT_TRANSCRIPT_OVERAGE_RATIO", "0.025")
            ),
            topic_working_set_enabled=_env_bool(
                "ATAGIA_TOPIC_WORKING_SET_ENABLED", True
            ),
            topic_working_set_refresh_message_lag=int(
                os.getenv("ATAGIA_TOPIC_WORKING_SET_REFRESH_MESSAGE_LAG", "4")
            ),
            topic_working_set_stale_message_lag=int(
                os.getenv("ATAGIA_TOPIC_WORKING_SET_STALE_MESSAGE_LAG", "10")
            ),
            topic_working_set_refresh_token_lag=int(
                os.getenv("ATAGIA_TOPIC_WORKING_SET_REFRESH_TOKEN_LAG", "2000")
            ),
            topic_working_set_stale_token_lag=int(
                os.getenv("ATAGIA_TOPIC_WORKING_SET_STALE_TOKEN_LAG", "5000")
            ),
            topic_working_set_refresh_batch_messages=int(
                os.getenv("ATAGIA_TOPIC_WORKING_SET_REFRESH_BATCH_MESSAGES", "8")
            ),
            graph_projection_enabled=_env_bool(
                "ATAGIA_GRAPH_PROJECTION_ENABLED", False
            ),
            verbatim_evidence_search_enabled=_env_bool(
                "ATAGIA_VERBATIM_EVIDENCE_SEARCH_ENABLED",
                True,
            ),
            verbatim_evidence_search_rrf_weight=float(
                os.getenv("ATAGIA_VERBATIM_EVIDENCE_SEARCH_RRF_WEIGHT", "0.75")
            ),
            verbatim_evidence_search_limit=int(
                os.getenv("ATAGIA_VERBATIM_EVIDENCE_SEARCH_LIMIT", "8")
            ),
            verbatim_evidence_window_size=int(
                os.getenv("ATAGIA_VERBATIM_EVIDENCE_WINDOW_SIZE", "3")
            ),
            verbatim_evidence_window_overlap=int(
                os.getenv("ATAGIA_VERBATIM_EVIDENCE_WINDOW_OVERLAP", "1")
            ),
            openai_proxy_model_id=os.getenv(
                "ATAGIA_PROXY_MODEL_ID",
                "atagia-memory-proxy",
            ),
            openai_proxy_upstream_model=_env_optional_str(
                "ATAGIA_PROXY_UPSTREAM_MODEL"
            ),
            openai_proxy_default_mode=_env_optional_str("ATAGIA_PROXY_DEFAULT_MODE"),
            cors_allowed_origins=_env_csv_tuple(
                "ATAGIA_CORS_ALLOWED_ORIGINS",
                (),
            ),
        )

    def effective_recent_transcript_budget_tokens(
        self,
        policy_budget_tokens: int,
        *,
        hard_cap_tokens: int | None = None,
    ) -> int:
        """Return the transcript budget used for immediate working memory."""
        configured_budget = (
            self.recent_transcript_budget_tokens
            if self.recent_transcript_budget_tokens is not None
            else policy_budget_tokens
        )
        effective_budget = max(
            MIN_RECENT_TRANSCRIPT_BUDGET_TOKENS, int(configured_budget)
        )
        if hard_cap_tokens is not None:
            effective_budget = min(effective_budget, int(hard_cap_tokens))
        return effective_budget

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
