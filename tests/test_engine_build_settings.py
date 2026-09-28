"""Tests for library-mode Settings assembly in ``Atagia._build_settings``.

Library mode is the mode every benchmark uses. ``_build_settings`` must forward
*all* env-configured Settings fields to the runtime while still applying the
engine's constructor-level overrides with their exact merge semantics. These
tests lock that contract:

* the env-derived base flows through for every field outside an explicit
  override allowlist (regression guard against silently dropping fields),
* the allowlist is the single source of truth shared between engine and tests,
  and
* omitting a constructor argument never discards the environment -- the
  allowlist says which fields the engine MAY write, and being on it is not
  permission to overwrite env with a hardcoded default.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from pathlib import Path

import pytest

from atagia import Atagia
from atagia.engine import (
    _ENGINE_FORCED_SETTINGS_FIELDS,
    _ENGINE_SETTINGS_OVERRIDE_FIELDS,
)
from atagia.core.config import Settings


def _settings_field_names() -> set[str]:
    return {field.name for field in dataclass_fields(Settings)}


def _env_for_overridable_fields(tmp_path: Path) -> dict[str, tuple[str, str]]:
    """One env var and value per allowlisted field library mode does not force.

    Every value differs from the code default on purpose: a constructor
    parameter that writes its own default over the environment is only
    detectable when the two disagree. That is exactly how ``sqlite_path``,
    ``skip_belief_revision`` and ``skip_compaction`` stayed broken while the
    introspection guard passed -- it skipped the whole allowlist.
    """
    resources = Path(__file__).resolve().parents[1] / "src" / "atagia" / "resources"
    return {
        "sqlite_path": ("ATAGIA_SQLITE_PATH", str(tmp_path / "env-sqlite.db")),
        "manifests_path": ("ATAGIA_MANIFESTS_PATH", str(resources / "manifests")),
        "operational_profiles_path": (
            "ATAGIA_OPERATIONAL_PROFILES_PATH",
            str(resources / "operational_profiles"),
        ),
        "storage_backend": ("ATAGIA_STORAGE_BACKEND", "redis"),
        "redis_url": ("ATAGIA_REDIS_URL", "redis://env.example:6380/3"),
        "anthropic_api_key": ("ATAGIA_ANTHROPIC_API_KEY", "env-anthropic-key"),
        "openai_api_key": ("ATAGIA_OPENAI_API_KEY", "env-openai-key"),
        "google_api_key": ("ATAGIA_GOOGLE_API_KEY", "env-google-key"),
        "openrouter_api_key": ("ATAGIA_OPENROUTER_API_KEY", "env-openrouter-key"),
        "inference_access_mode": ("ATAGIA_INFERENCE_ACCESS_MODE", "zero_cost"),
        "local_llm_endpoints_file": (
            "ATAGIA_LOCAL_LLM_ENDPOINTS_FILE",
            str(tmp_path / "local-endpoints.json"),
        ),
        "zero_cost_openrouter_profile": (
            "ATAGIA_ZERO_COST_OPENROUTER_PROFILE",
            "dedicated_free_tier_no_byok",
        ),
        "llm_chat_model": ("ATAGIA_LLM_CHAT_MODEL", "openai/env-chat"),
        "llm_forced_global_model": (
            "ATAGIA_LLM_FORCED_GLOBAL_MODEL",
            "openai/env-forced",
        ),
        "llm_ingest_model": ("ATAGIA_LLM_INGEST_MODEL", "openai/env-ingest"),
        "llm_retrieval_model": ("ATAGIA_LLM_RETRIEVAL_MODEL", "openai/env-retrieval"),
        "llm_component_models": ("ATAGIA_LLM_MODEL__EXTRACTOR", "openai/env-extractor"),
        "llm_intimacy_ingest_model": (
            "ATAGIA_LLM_INTIMACY_INGEST_MODEL",
            "openai/env-intimacy-ingest",
        ),
        "llm_intimacy_retrieval_model": (
            "ATAGIA_LLM_INTIMACY_RETRIEVAL_MODEL",
            "openai/env-intimacy-retrieval",
        ),
        "llm_intimacy_component_models": (
            "ATAGIA_LLM_INTIMACY_MODEL__EXTRACTOR",
            "openai/env-intimacy-extractor",
        ),
        "llm_intimacy_proactive_routing_enabled": (
            "ATAGIA_LLM_INTIMACY_PROACTIVE_ROUTING_ENABLED",
            "true",
        ),
        "llm_structured_output_retry_attempts": (
            "ATAGIA_LLM_STRUCTURED_OUTPUT_RETRY_ATTEMPTS",
            "4",
        ),
        "llm_structured_output_rescue_enabled": (
            "ATAGIA_LLM_STRUCTURED_OUTPUT_RESCUE_ENABLED",
            "true",
        ),
        "llm_structured_output_rescue_model": (
            "ATAGIA_LLM_STRUCTURED_OUTPUT_RESCUE_MODEL",
            "openai/env-rescue",
        ),
        "answer_postcondition_guard_enabled": (
            "ATAGIA_ANSWER_POSTCONDITION_GUARD_ENABLED",
            "true",
        ),
        "answer_stance": ("ATAGIA_ANSWER_STANCE", "proactive"),
        "answer_stance_prompt_variant": (
            "ATAGIA_ANSWER_STANCE_PROMPT_VARIANT",
            "template_v1",
        ),
        "embedding_backend": ("ATAGIA_EMBEDDING_BACKEND", "sqlite_vec"),
        "embedding_model": ("ATAGIA_EMBEDDING_MODEL", "openai/env-embedding"),
        "skip_belief_revision": ("ATAGIA_SKIP_BELIEF_REVISION", "true"),
        "skip_compaction": ("ATAGIA_SKIP_COMPACTION", "true"),
        "context_cache_enabled": ("ATAGIA_CONTEXT_CACHE_ENABLED", "false"),
        "disable_chunking_extraction": ("ATAGIA_DISABLE_CHUNKING_EXTRACTION", "true"),
        "assistant_guidance_enabled": ("ATAGIA_ASSISTANT_GUIDANCE_ENABLED", "false"),
        "context_envelope_budget_tokens": (
            "ATAGIA_CONTEXT_ENVELOPE_BUDGET_TOKENS",
            "31337",
        ),
        "context_envelope_ratios": (
            "ATAGIA_CONTEXT_ENVELOPE_RATIOS",
            '{"instructions": 0.2, "current_turn": 0.05, '
            '"retrieved_context": 0.55, "recent_transcript": 0.2}',
        ),
    }


def test_override_allowlist_is_a_valid_settings_subset() -> None:
    """The allowlist must reference only real Settings fields and exclude
    fields the engine forwards straight from env."""
    field_names = _settings_field_names()

    assert _ENGINE_SETTINGS_OVERRIDE_FIELDS
    assert _ENGINE_SETTINGS_OVERRIDE_FIELDS <= field_names

    # Sanity: fields known to be pure env passthroughs must not be listed as
    # engine overrides (these are exactly the kind of field the old constructor
    # silently dropped).
    for passthrough in (
        "rrf_k",
        "llm_request_timeout_seconds",
        "card_examples_enabled",
        "llm_runaway_max_checks",
        "default_language_code",
    ):
        assert passthrough in field_names
        assert passthrough not in _ENGINE_SETTINGS_OVERRIDE_FIELDS


def test_build_settings_forwards_every_env_field_outside_allowlist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Introspection invariant: every Settings field is either an explicit
    engine override or equals the env-derived value. Setting env vars that
    differ from the dataclass defaults across many previously-dropped families
    turns this into a real regression guard: under the old explicit-kwargs
    constructor these fields fell back to defaults instead of env.

    This test also cross-checks the single source of truth: ``_build_settings``
    raises if its override dict keys drift from ``_ENGINE_SETTINGS_OVERRIDE_FIELDS``,
    so merely building settings here exercises that guard.
    """
    curated_env = {
        # Fields that used to be dropped, spanning several families and types.
        "ATAGIA_RRF_K": "99",
        "ATAGIA_RECALL_RECOVERY_SCORING_MAX_CANDIDATES": "32",
        "ATAGIA_BROAD_LIST_COMPARABLE_RRF_ENABLED": "true",
        "ATAGIA_FUSED_CANDIDATE_GUARD_ORDERING_ENABLED": "true",
        "ATAGIA_LLM_REQUEST_TIMEOUT_SECONDS": "45.5",
        "ATAGIA_ANTHROPIC_REQUEST_TIMEOUT_SECONDS": "88.0",
        "ATAGIA_LLM_RUNAWAY_WATCHDOG_ENABLED": "false",
        "ATAGIA_LLM_RUNAWAY_MAX_CHECKS": "7",
        "ATAGIA_LLM_RUNAWAY_OUTPUT_INPUT_RATIO": "15.0",
        "ATAGIA_CARD_EXAMPLES_ENABLED": "false",
        "ATAGIA_DEFAULT_LANGUAGE_CODE": "es",
        "ATAGIA_WORKER_TRANSIENT_DEFER_SECONDS": "90.0",
        "ATAGIA_WORKER_DISPATCH_BATCH_SIZE": "37",
        "ATAGIA_BELIEF_TENSION_THRESHOLD": "0.75",
        "ATAGIA_EPISODE_SYNTHESIS_MAX_EPISODES": "12",
        "ATAGIA_OPF_TIMEOUT_SECONDS": "5.0",
        "ATAGIA_PRIVACY_VALIDATION_GATE_ENABLED": "true",
        "ATAGIA_INITIAL_CONTEXT_PACKAGE_TOTAL_MAX_TOKENS": "3000",
        "ATAGIA_REQUEST_MAX_METADATA_BYTES": "131072",
        "ATAGIA_CORS_ALLOWED_ORIGINS": "https://example.com",
        "ATAGIA_LLM_OUTPUT_LIMIT_RETRY_ATTEMPTS": "3",
        "ATAGIA_LLM_TECHNICAL_RECOVERY_ENABLED": "false",
        "ATAGIA_PROXY_MAX_OUTPUT_TOKENS": "9000",
    }
    for name, value in curated_env.items():
        monkeypatch.setenv(name, value)

    env_settings = Settings.from_env()
    engine = Atagia(db_path=tmp_path / "introspection.db")
    built = engine._build_settings()

    for field in dataclass_fields(Settings):
        if field.name in _ENGINE_SETTINGS_OVERRIDE_FIELDS:
            continue
        assert getattr(built, field.name) == getattr(env_settings, field.name), (
            f"field {field.name!r} is not an engine override yet diverged from "
            "the env-derived value"
        )

    # Explicit checks on previously-dropped fields so the enumeration above is a
    # genuine regression guard (these all differ from the dataclass defaults).
    assert built.rrf_k == 99
    assert built.recall_recovery_scoring_max_candidates == 32
    assert built.broad_list_comparable_rrf_enabled is True
    assert built.fused_candidate_guard_ordering_enabled is True
    assert built.llm_request_timeout_seconds == 45.5
    assert built.anthropic_request_timeout_seconds == 88.0
    assert built.card_examples_enabled is False
    assert built.default_language_code == "es"
    assert built.worker_transient_defer_seconds == 90.0
    assert built.worker_dispatch_batch_size == 37
    assert built.belief_tension_threshold == 0.75
    assert built.cors_allowed_origins == ("https://example.com",)
    assert built.llm_runaway_watchdog_enabled is False
    assert built.llm_output_limit_retry_attempts == 3


def test_engine_forced_fields_are_exactly_the_service_mode_pins() -> None:
    """The forced set is the escape hatch from the env-must-arrive rule, so it
    is pinned literally: growing it is how a field silently stops being
    configurable, and it must be a deliberate, reviewed edit.

    Every member is a hardcoded literal in ``_build_settings`` whose env var is
    service-mode configuration read by ``app.py``, never library-mode
    configuration.
    """
    assert _ENGINE_FORCED_SETTINGS_FIELDS == frozenset(
        {
            "service_mode",
            "service_api_key",
            "admin_api_key",
            "workers_enabled",
            "allow_insecure_http",
        }
    )


def test_omitting_a_constructor_argument_never_discards_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An engine built with NO arguments must run on the environment.

    This is the introspection guard the allowlist used to hide from: it skipped
    every allowlisted field, so a constructor parameter defaulting to a real
    value ("atagia.db", False) rather than ``None`` was indistinguishable from
    the caller passing that value, and ``ATAGIA_SQLITE_PATH`` /
    ``ATAGIA_SKIP_*`` were dropped on every boot with the suite green. Only the
    five service-mode pins may diverge from env here.
    """
    field_env = _env_for_overridable_fields(tmp_path)
    overridable = _ENGINE_SETTINGS_OVERRIDE_FIELDS - _ENGINE_FORCED_SETTINGS_FIELDS
    assert set(field_env) == overridable, (
        "every allowlisted field the engine does not force needs env coverage "
        "here, otherwise a new one can silently discard its variable again"
    )
    for env_var, value in field_env.values():
        monkeypatch.setenv(env_var, value)

    env_settings = Settings.from_env()
    built = Atagia()._build_settings()

    for field in dataclass_fields(Settings):
        if field.name in _ENGINE_FORCED_SETTINGS_FIELDS:
            continue
        assert getattr(built, field.name) == getattr(env_settings, field.name), (
            f"field {field.name!r} lost its env-derived value even though no "
            "constructor argument was supplied"
        )

    # Spot checks on the three the defect actually hit, so the enumeration above
    # cannot pass vacuously.
    assert built.sqlite_path == str(tmp_path / "env-sqlite.db")
    assert built.skip_belief_revision is True
    assert built.skip_compaction is True


def test_build_settings_carries_previously_dropped_env_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Functional check on the exact families called out by the defect:
    rrf_k, an llm_runaway_* field, and the llm request timeout."""
    monkeypatch.setenv("ATAGIA_RRF_K", "123")
    monkeypatch.setenv("ATAGIA_LLM_RUNAWAY_MIN_OUTPUT_TOKENS", "4096")
    monkeypatch.setenv("ATAGIA_LLM_REQUEST_TIMEOUT_SECONDS", "73.5")
    engine = Atagia(db_path=tmp_path / "dropped-env.db")

    settings = engine._build_settings()

    assert settings.rrf_k == 123
    assert settings.llm_runaway_min_output_tokens == 4096
    assert settings.llm_request_timeout_seconds == 73.5


def test_build_settings_constructor_override_wins_over_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """For overridable fields, an explicit constructor argument wins over env."""
    monkeypatch.setenv("ATAGIA_CONTEXT_CACHE_ENABLED", "false")
    monkeypatch.setenv("ATAGIA_ANSWER_STANCE", "reactive")
    monkeypatch.setenv("ATAGIA_INFERENCE_ACCESS_MODE", "unrestricted")
    monkeypatch.setenv(
        "ATAGIA_LOCAL_LLM_ENDPOINTS_FILE",
        str(tmp_path / "env-catalog.json"),
    )
    engine = Atagia(
        db_path=tmp_path / "override-wins.db",
        context_cache_enabled=True,
        answer_stance="proactive",
        inference_access_mode="local_only",
        local_llm_endpoints_file=tmp_path / "constructor-catalog.json",
    )

    settings = engine._build_settings()

    assert settings.context_cache_enabled is True
    assert settings.answer_stance == "proactive"
    assert settings.inference_access_mode == "local_only"
    assert settings.local_llm_endpoints_file == str(
        tmp_path / "constructor-catalog.json"
    )


def test_build_settings_env_wins_when_constructor_arg_absent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """For overridable fields, env wins when the constructor argument is absent
    (left at its ``None`` default)."""
    monkeypatch.setenv("ATAGIA_CONTEXT_CACHE_ENABLED", "false")
    monkeypatch.setenv("ATAGIA_ANSWER_STANCE", "proactive")
    engine = Atagia(db_path=tmp_path / "env-wins.db")

    settings = engine._build_settings()

    assert settings.context_cache_enabled is False
    assert settings.answer_stance == "proactive"


def test_build_settings_applies_engine_hardcoded_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Library mode pins a few fields regardless of env (service is off,
    workers on, insecure HTTP allowed)."""
    monkeypatch.setenv("ATAGIA_WORKERS_ENABLED", "false")
    monkeypatch.setenv("ATAGIA_SERVICE_MODE", "true")
    monkeypatch.setenv("ATAGIA_ALLOW_INSECURE_HTTP", "false")
    engine = Atagia(db_path=tmp_path / "hardcoded.db")

    settings = engine._build_settings()

    assert settings.workers_enabled is True
    assert settings.service_mode is False
    assert settings.service_api_key is None
    assert settings.admin_api_key is None
    assert settings.allow_insecure_http is True


def test_build_settings_guards_against_override_allowlist_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The runtime guard fails fast if the override dict keys stop matching the
    single-source allowlist constant."""
    monkeypatch.setattr(
        "atagia.engine._ENGINE_SETTINGS_OVERRIDE_FIELDS",
        frozenset({"sqlite_path"}),
    )
    engine = Atagia(db_path=tmp_path / "drift.db")

    with pytest.raises(RuntimeError, match="override keys drifted"):
        engine._build_settings()


def test_build_settings_guard_fires_on_extra_allowlist_entry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The guard also fires in the opposite direction: an allowlist entry that
    the override dict does not actually build (extra key) is drift too."""
    from atagia.engine import _ENGINE_SETTINGS_OVERRIDE_FIELDS

    monkeypatch.setattr(
        "atagia.engine._ENGINE_SETTINGS_OVERRIDE_FIELDS",
        _ENGINE_SETTINGS_OVERRIDE_FIELDS | {"rrf_k"},
    )
    engine = Atagia(db_path=tmp_path / "drift_extra.db")

    with pytest.raises(RuntimeError, match="override keys drifted"):
        engine._build_settings()


def test_engine_override_fields_are_a_subset_of_the_allowlist(
    tmp_path: Path,
) -> None:
    """Provenance reports a per-boot SUBSET of the structural allowlist: the
    fields library mode pins plus the ones this caller supplied. A field the
    engine only copied back from env is not in it."""
    engine = Atagia(
        db_path=tmp_path / "override-subset.db",
        answer_stance="proactive",
    )

    engine._build_settings()

    assert engine._engine_override_fields <= _ENGINE_SETTINGS_OVERRIDE_FIELDS
    assert _ENGINE_FORCED_SETTINGS_FIELDS <= engine._engine_override_fields
    assert "answer_stance" in engine._engine_override_fields
    assert "llm_chat_model" not in engine._engine_override_fields
    assert engine._engine_override_fields < _ENGINE_SETTINGS_OVERRIDE_FIELDS


def test_build_settings_guards_against_unclassified_override_field(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Every allowlisted field must be classified for provenance. Dropping one
    from the engine-forced set without giving it a merge classification leaves
    it unattributable, and the guard fires instead of silently reporting it as
    ``env``."""
    monkeypatch.setattr(
        "atagia.engine._ENGINE_FORCED_SETTINGS_FIELDS",
        _ENGINE_FORCED_SETTINGS_FIELDS - {"workers_enabled"},
    )
    engine = Atagia(db_path=tmp_path / "unclassified.db")

    with pytest.raises(RuntimeError, match="provenance classification drifted"):
        engine._build_settings()


def test_build_settings_guards_against_double_classified_override_field(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The classification is a partition: a field claimed by two groups at once
    (here engine-forced AND the ``redis_url`` fallback merge) would make its
    provenance depend on evaluation order, so the guard fires."""
    monkeypatch.setattr(
        "atagia.engine._ENGINE_FORCED_SETTINGS_FIELDS",
        _ENGINE_FORCED_SETTINGS_FIELDS | {"redis_url"},
    )
    engine = Atagia(db_path=tmp_path / "double-classified.db")

    with pytest.raises(RuntimeError, match="provenance classification drifted"):
        engine._build_settings()
