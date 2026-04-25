"""Configuration loading from environment variables."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


_DOTENV_LOADED = False


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


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_csv_tuple(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = tuple(part.strip() for part in value.split(",") if part.strip())
    return normalized or default


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings for Atagia."""

    sqlite_path: str
    migrations_path: str
    manifests_path: str
    storage_backend: str
    redis_url: str
    llm_provider: str
    llm_api_key: str | None
    openai_api_key: str | None
    openrouter_api_key: str | None
    llm_base_url: str | None
    openrouter_site_url: str
    openrouter_app_name: str
    llm_extraction_model: str | None
    llm_scoring_model: str | None
    llm_classifier_model: str | None
    llm_chat_model: str | None
    service_mode: bool
    service_api_key: str | None
    admin_api_key: str | None
    workers_enabled: bool
    debug: bool
    operational_profiles_path: str = "./operational_profiles"
    allow_admin_export_anonymization: bool = False
    allow_insecure_http: bool = False
    embedding_provider_name: str | None = None
    embedding_backend: str = "none"
    embedding_model: str | None = None
    embedding_dimension: int = 1536
    rrf_k: int = 60
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
    opf_privacy_filter_enabled: bool = False
    opf_primary_url: str = "http://127.0.0.1:8008"
    opf_fallback_url: str = "http://127.0.0.1:8008"
    opf_timeout_seconds: float = 2.0
    privacy_validation_gate_enabled: bool = False
    privacy_validation_gate_timeout_seconds: float = 20.0
    privacy_validation_gate_max_source_chars: int = 6000
    privacy_validation_gate_max_summaries_gated_per_job: int = 50
    privacy_validation_gate_judge_model: str | None = None
    privacy_validation_gate_refiner_model: str | None = None
    operational_high_risk_enabled: bool = False
    operational_allowed_profiles: tuple[str, ...] = ("normal", "low_power", "offline")
    context_cache_enabled: bool = True
    context_cache_min_ttl_seconds: int = 60
    context_cache_max_ttl_seconds: int = 3600
    chunking_enabled: bool = False
    chunking_threshold_tokens: int = 2000
    lifecycle_lazy_enabled: bool = True
    lifecycle_min_interval_seconds: int = 3600
    lifecycle_worker_enabled: bool = False
    lifecycle_worker_interval_seconds: int = 3600
    small_corpus_token_threshold_ratio: float = 0.7
    # Wave 1 batch 2 (1-C): raw evidence as first-class search channel.
    raw_message_channel: bool = True
    # Raw evidence RRF weight, slightly lower than memory_objects by
    # default because memories are more focused and raw evidence is a
    # recall safety net. Tuning is deferred to Wave 2.
    raw_message_rrf_weight: float = 0.75
    # Maximum conversation windows fetched per sub-query from the raw
    # evidence channel before fusion. Raw evidence is a secondary lane
    # so this ceiling is modest.
    raw_message_channel_limit: int = 8
    # Conversation window size (messages per window) used by the raw
    # evidence channel. 2-4 per the Wave 1-C spec; default 3 with 1-turn
    # overlap keeps neighbouring evidence accessible.
    raw_message_window_size: int = 3
    raw_message_window_overlap: int = 1

    def __post_init__(self) -> None:
        if self.context_cache_min_ttl_seconds <= 0:
            raise ValueError("context_cache_min_ttl_seconds must be positive")
        if self.context_cache_max_ttl_seconds <= 0:
            raise ValueError("context_cache_max_ttl_seconds must be positive")
        if self.context_cache_max_ttl_seconds < self.context_cache_min_ttl_seconds:
            raise ValueError("context_cache_max_ttl_seconds must be >= context_cache_min_ttl_seconds")
        if self.chunking_threshold_tokens <= 0:
            raise ValueError("chunking_threshold_tokens must be positive")
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if self.lifecycle_min_interval_seconds <= 0:
            raise ValueError("lifecycle_min_interval_seconds must be positive")
        if self.lifecycle_worker_enabled and self.lifecycle_worker_interval_seconds <= 0:
            raise ValueError("lifecycle_worker_interval_seconds must be positive")
        if self.belief_tension_increment <= 0:
            raise ValueError("belief_tension_increment must be positive")
        if self.belief_tension_decrement <= 0:
            raise ValueError("belief_tension_decrement must be positive")
        if self.belief_tension_threshold < 0:
            raise ValueError("belief_tension_threshold must be non-negative")
        if self.ephemeral_scoring_hours <= 0:
            raise ValueError("ephemeral_scoring_hours must be positive")
        if self.opf_timeout_seconds <= 0:
            raise ValueError("opf_timeout_seconds must be positive")
        if self.privacy_validation_gate_timeout_seconds <= 0:
            raise ValueError("privacy_validation_gate_timeout_seconds must be positive")
        if self.privacy_validation_gate_max_source_chars <= 0:
            raise ValueError("privacy_validation_gate_max_source_chars must be positive")
        if self.privacy_validation_gate_max_summaries_gated_per_job < 0:
            raise ValueError("privacy_validation_gate_max_summaries_gated_per_job must be non-negative")
        if not self.operational_allowed_profiles:
            raise ValueError("operational_allowed_profiles must contain at least one profile")
        if any(not profile_id.strip() for profile_id in self.operational_allowed_profiles):
            raise ValueError("operational_allowed_profiles cannot contain blank profile ids")
        if not 0.0 <= self.small_corpus_token_threshold_ratio <= 1.0:
            raise ValueError(
                "small_corpus_token_threshold_ratio must be in the interval [0.0, 1.0]"
            )
        if not 0.0 <= self.raw_message_rrf_weight <= 2.0:
            raise ValueError("raw_message_rrf_weight must be in the interval [0.0, 2.0]")
        if self.raw_message_channel_limit < 0:
            raise ValueError("raw_message_channel_limit must be non-negative")
        if self.raw_message_window_size < 2 or self.raw_message_window_size > 4:
            raise ValueError("raw_message_window_size must be between 2 and 4")
        if self.raw_message_window_overlap < 0:
            raise ValueError("raw_message_window_overlap must be non-negative")
        if self.raw_message_window_overlap >= self.raw_message_window_size:
            raise ValueError(
                "raw_message_window_overlap must be strictly less than raw_message_window_size"
            )

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv_once()
        return cls(
            sqlite_path=os.getenv("ATAGIA_SQLITE_PATH", "./data/atagia.db"),
            migrations_path=os.getenv("ATAGIA_MIGRATIONS_PATH", "./migrations"),
            manifests_path=os.getenv("ATAGIA_MANIFESTS_PATH", "./manifests"),
            operational_profiles_path=os.getenv(
                "ATAGIA_OPERATIONAL_PROFILES_PATH",
                "./operational_profiles",
            ),
            storage_backend=os.getenv("ATAGIA_STORAGE_BACKEND", "inprocess").strip().lower(),
            redis_url=os.getenv("ATAGIA_REDIS_URL", "redis://localhost:6379/0"),
            llm_provider=os.getenv("ATAGIA_LLM_PROVIDER", "anthropic").strip().lower(),
            llm_api_key=os.getenv("ATAGIA_LLM_API_KEY") or None,
            openai_api_key=os.getenv("ATAGIA_OPENAI_API_KEY") or None,
            openrouter_api_key=os.getenv("ATAGIA_OPENROUTER_API_KEY") or None,
            llm_base_url=os.getenv("ATAGIA_LLM_BASE_URL") or None,
            openrouter_site_url=os.getenv("ATAGIA_OPENROUTER_SITE_URL", "http://localhost"),
            openrouter_app_name=os.getenv("ATAGIA_OPENROUTER_APP_NAME", "Atagia"),
            llm_extraction_model=os.getenv("ATAGIA_LLM_EXTRACTION_MODEL") or None,
            llm_scoring_model=os.getenv("ATAGIA_LLM_SCORING_MODEL") or None,
            llm_classifier_model=(
                os.getenv("ATAGIA_LLM_CLASSIFIER_MODEL")
                or os.getenv("ATAGIA_LLM_SCORING_MODEL")
                or None
            ),
            llm_chat_model=os.getenv("ATAGIA_LLM_CHAT_MODEL") or None,
            embedding_provider_name=os.getenv("ATAGIA_EMBEDDING_PROVIDER") or None,
            service_mode=_env_bool("ATAGIA_SERVICE_MODE", False),
            service_api_key=os.getenv("ATAGIA_SERVICE_API_KEY") or None,
            admin_api_key=os.getenv("ATAGIA_ADMIN_API_KEY") or None,
            allow_admin_export_anonymization=_env_bool(
                "ATAGIA_ALLOW_ADMIN_EXPORT_ANONYMIZATION",
                False,
            ),
            workers_enabled=_env_bool("ATAGIA_WORKERS_ENABLED", False),
            debug=_env_bool("ATAGIA_DEBUG", False),
            allow_insecure_http=_env_bool("ATAGIA_ALLOW_INSECURE_HTTP", False),
            embedding_backend=os.getenv("ATAGIA_EMBEDDING_BACKEND", "none").strip().lower(),
            embedding_model=os.getenv("ATAGIA_EMBEDDING_MODEL") or None,
            embedding_dimension=int(os.getenv("ATAGIA_EMBEDDING_DIMENSION", "1536")),
            rrf_k=int(os.getenv("ATAGIA_RRF_K", "60")),
            lifecycle_decay_days=int(os.getenv("ATAGIA_LIFECYCLE_DECAY_DAYS", "7")),
            lifecycle_decay_rate=float(os.getenv("ATAGIA_LIFECYCLE_DECAY_RATE", "0.9")),
            lifecycle_archive_vitality=float(os.getenv("ATAGIA_LIFECYCLE_ARCHIVE_VITALITY", "0.05")),
            lifecycle_archive_confidence=float(os.getenv("ATAGIA_LIFECYCLE_ARCHIVE_CONFIDENCE", "0.3")),
            ephemeral_scoring_hours=int(os.getenv("ATAGIA_EPHEMERAL_SCORING_HOURS", "24")),
            lifecycle_ephemeral_ttl_hours=int(os.getenv("ATAGIA_LIFECYCLE_EPHEMERAL_TTL_HOURS", "24")),
            lifecycle_review_ttl_days=int(os.getenv("ATAGIA_LIFECYCLE_REVIEW_TTL_DAYS", "7")),
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
            belief_tension_increment=float(os.getenv("ATAGIA_BELIEF_TENSION_INCREMENT", "0.15")),
            belief_tension_decrement=float(os.getenv("ATAGIA_BELIEF_TENSION_DECREMENT", "0.05")),
            belief_tension_threshold=float(os.getenv("ATAGIA_BELIEF_TENSION_THRESHOLD", "0.5")),
            skip_belief_revision=_env_bool("ATAGIA_SKIP_BELIEF_REVISION", False),
            skip_compaction=_env_bool("ATAGIA_SKIP_COMPACTION", False),
            opf_privacy_filter_enabled=_env_bool("ATAGIA_OPF_PRIVACY_FILTER_ENABLED", False),
            opf_primary_url=os.getenv("ATAGIA_OPF_PRIMARY_URL", "http://127.0.0.1:8008"),
            opf_fallback_url=os.getenv("ATAGIA_OPF_FALLBACK_URL", "http://127.0.0.1:8008"),
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
                os.getenv("ATAGIA_PRIVACY_VALIDATION_GATE_MAX_SUMMARIES_GATED_PER_JOB", "50")
            ),
            privacy_validation_gate_judge_model=(
                os.getenv("ATAGIA_PRIVACY_VALIDATION_GATE_JUDGE_MODEL") or None
            ),
            privacy_validation_gate_refiner_model=(
                os.getenv("ATAGIA_PRIVACY_VALIDATION_GATE_REFINER_MODEL") or None
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
            context_cache_min_ttl_seconds=int(
                os.getenv("ATAGIA_CONTEXT_CACHE_MIN_TTL_SECONDS", "60")
            ),
            context_cache_max_ttl_seconds=int(
                os.getenv("ATAGIA_CONTEXT_CACHE_MAX_TTL_SECONDS", "3600")
            ),
            chunking_enabled=_env_bool("ATAGIA_CHUNKING_ENABLED", False),
            chunking_threshold_tokens=int(
                os.getenv("ATAGIA_CHUNKING_THRESHOLD_TOKENS", "2000")
            ),
            lifecycle_lazy_enabled=_env_bool("ATAGIA_LIFECYCLE_LAZY_ENABLED", True),
            lifecycle_min_interval_seconds=int(
                os.getenv("ATAGIA_LIFECYCLE_MIN_INTERVAL_SECONDS", "3600")
            ),
            lifecycle_worker_enabled=_env_bool("ATAGIA_LIFECYCLE_WORKER_ENABLED", False),
            lifecycle_worker_interval_seconds=int(
                os.getenv("ATAGIA_LIFECYCLE_WORKER_INTERVAL_SECONDS", "3600")
            ),
            small_corpus_token_threshold_ratio=float(
                os.getenv("ATAGIA_SMALL_CORPUS_TOKEN_THRESHOLD_RATIO", "0.7")
            ),
            raw_message_channel=_env_bool("ATAGIA_RAW_MESSAGE_CHANNEL", True),
            raw_message_rrf_weight=float(
                os.getenv("ATAGIA_RAW_MESSAGE_RRF_WEIGHT", "0.75")
            ),
            raw_message_channel_limit=int(
                os.getenv("ATAGIA_RAW_MESSAGE_CHANNEL_LIMIT", "8")
            ),
            raw_message_window_size=int(
                os.getenv("ATAGIA_RAW_MESSAGE_WINDOW_SIZE", "3")
            ),
            raw_message_window_overlap=int(
                os.getenv("ATAGIA_RAW_MESSAGE_WINDOW_OVERLAP", "1")
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
