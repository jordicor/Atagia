"""Configuration loading from environment variables."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
    allow_insecure_http: bool = False
    embedding_backend: str = "none"
    embedding_model: str | None = None
    embedding_dimension: int = 1536
    lifecycle_decay_days: int = 7
    lifecycle_decay_rate: float = 0.9
    lifecycle_archive_vitality: float = 0.05
    lifecycle_archive_confidence: float = 0.3
    lifecycle_ephemeral_ttl_hours: int = 24
    lifecycle_review_ttl_days: int = 7
    promotion_conv_to_ws_min_conversations: int = 2
    promotion_ws_to_global_min_sessions: int = 3
    promotion_require_mode_consistency: bool = True
    skip_belief_revision: bool = False
    skip_compaction: bool = False
    context_cache_enabled: bool = True
    context_cache_min_ttl_seconds: int = 60
    context_cache_max_ttl_seconds: int = 3600
    chunking_enabled: bool = False
    chunking_threshold_tokens: int = 2000

    def __post_init__(self) -> None:
        if self.context_cache_min_ttl_seconds <= 0:
            raise ValueError("context_cache_min_ttl_seconds must be positive")
        if self.context_cache_max_ttl_seconds <= 0:
            raise ValueError("context_cache_max_ttl_seconds must be positive")
        if self.context_cache_max_ttl_seconds < self.context_cache_min_ttl_seconds:
            raise ValueError("context_cache_max_ttl_seconds must be >= context_cache_min_ttl_seconds")
        if self.chunking_threshold_tokens <= 0:
            raise ValueError("chunking_threshold_tokens must be positive")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            sqlite_path=os.getenv("ATAGIA_SQLITE_PATH", "./data/atagia.db"),
            migrations_path=os.getenv("ATAGIA_MIGRATIONS_PATH", "./migrations"),
            manifests_path=os.getenv("ATAGIA_MANIFESTS_PATH", "./manifests"),
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
            service_mode=_env_bool("ATAGIA_SERVICE_MODE", False),
            service_api_key=os.getenv("ATAGIA_SERVICE_API_KEY") or None,
            admin_api_key=os.getenv("ATAGIA_ADMIN_API_KEY") or None,
            workers_enabled=_env_bool("ATAGIA_WORKERS_ENABLED", False),
            debug=_env_bool("ATAGIA_DEBUG", False),
            allow_insecure_http=_env_bool("ATAGIA_ALLOW_INSECURE_HTTP", False),
            embedding_backend=os.getenv("ATAGIA_EMBEDDING_BACKEND", "none").strip().lower(),
            embedding_model=os.getenv("ATAGIA_EMBEDDING_MODEL") or None,
            embedding_dimension=int(os.getenv("ATAGIA_EMBEDDING_DIMENSION", "1536")),
            lifecycle_decay_days=int(os.getenv("ATAGIA_LIFECYCLE_DECAY_DAYS", "7")),
            lifecycle_decay_rate=float(os.getenv("ATAGIA_LIFECYCLE_DECAY_RATE", "0.9")),
            lifecycle_archive_vitality=float(os.getenv("ATAGIA_LIFECYCLE_ARCHIVE_VITALITY", "0.05")),
            lifecycle_archive_confidence=float(os.getenv("ATAGIA_LIFECYCLE_ARCHIVE_CONFIDENCE", "0.3")),
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
            skip_belief_revision=_env_bool("ATAGIA_SKIP_BELIEF_REVISION", False),
            skip_compaction=_env_bool("ATAGIA_SKIP_COMPACTION", False),
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
        )

    def migrations_dir(self, root: Path | None = None) -> Path:
        base = root or Path.cwd()
        return (base / self.migrations_path).resolve()

    def manifests_dir(self, root: Path | None = None) -> Path:
        base = root or Path.cwd()
        return (base / self.manifests_path).resolve()
