"""Tests for the engine-level effective-settings report (provenance + redaction).

The report must prove what a run executed with: every ``Settings`` field with a
provenance tag derived from its real source (``default`` / ``env`` /
``engine_override``), the resolved retrieval policy with each field tagged by the
layer that produced it (``manifest`` / ``default`` / ``resolver`` /
``computed``), and secrets redacted mechanically so a credential value never
appears.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import pytest

from atagia import Atagia
from atagia.core.config import Settings
from atagia.core.effective_settings import (
    REDACTED_EMPTY,
    REDACTED_SET,
    ResolvedPolicyReport,
    build_effective_settings_report,
    is_secret_field,
)
from atagia.core.settings_env_vars import SETTINGS_ENV_VARS
from atagia.engine import (
    _ENGINE_FORCED_SETTINGS_FIELDS,
    _ENGINE_SETTINGS_OVERRIDE_FIELDS,
)
from atagia.services.errors import RuntimeNotInitializedError


def _settings_field_names() -> set[str]:
    return {field.name for field in dataclasses.fields(Settings)}


def _engine(db_path: Path) -> Atagia:
    """An engine that boots without network: one forced provider, one stub key."""
    return Atagia(
        db_path=db_path,
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )


def test_report_covers_every_settings_field() -> None:
    """Introspection invariant: the settings block has exactly one entry per
    Settings field, each with the value/provenance/redacted shape."""
    report = build_effective_settings_report(
        effective_settings=Settings.from_env(),
        present_env_names=frozenset(os.environ),
        engine_override_fields=_ENGINE_SETTINGS_OVERRIDE_FIELDS,
        resolved_policies={},
    )

    assert set(report) == {"settings", "resolved_policy"}
    assert set(report["settings"]) == _settings_field_names()
    for entry in report["settings"].values():
        assert set(entry) == {"value", "provenance", "redacted"}
        assert entry["provenance"] in {"default", "env", "engine_override"}


@pytest.mark.asyncio
async def test_engine_setup_reports_provider_dispatch_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ATAGIA_LLM_MAX_CONCURRENT_REQUESTS_PER_PROVIDER", "7")

    async with _engine(tmp_path / "dispatch-limit.db") as engine:
        assert engine.runtime is not None
        assert engine.runtime.settings.llm_max_concurrent_requests_per_provider == 7
        assert engine.effective_settings_report()["settings"][
            "llm_max_concurrent_requests_per_provider"
        ] == {"value": 7, "provenance": "env", "redacted": False}


def test_is_secret_field_matches_credentials_not_token_counts() -> None:
    """Every ``*_api_key`` credential redacts; token-COUNT settings never do."""
    for name in _settings_field_names():
        if name.endswith("_api_key"):
            assert is_secret_field(name), name

    for count_field in (
        "context_envelope_budget_tokens",
        "llm_run_guard_max_total_tokens",
        "small_corpus_token_threshold_ratio",
        "topic_working_set_refresh_token_lag",
        "topic_working_set_stale_token_lag",
    ):
        assert count_field in _settings_field_names()
        assert not is_secret_field(count_field), count_field

    # Generic credential-shaped names still match the four markers.
    for credential in ("service_token", "client_secret", "db_password", "vault_api_key"):
        assert is_secret_field(credential), credential


@pytest.mark.asyncio
async def test_provenance_default_env_and_engine_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """One field per provenance class through the real engine accessor:
    an env-set passthrough (``env``), an untouched field (``default``), and an
    engine-forced field (``engine_override``)."""
    monkeypatch.setenv("ATAGIA_RRF_K", "77")  # differs from default 60
    monkeypatch.setenv("ATAGIA_SERVICE_MODE", "true")  # engine forces False
    monkeypatch.delenv("ATAGIA_LIFECYCLE_DECAY_DAYS", raising=False)

    async with _engine(tmp_path / "provenance.db") as engine:
        report = engine.effective_settings_report()["settings"]

    assert report["rrf_k"]["value"] == 77
    assert report["rrf_k"]["provenance"] == "env"

    assert report["lifecycle_decay_days"]["value"] == 7
    assert report["lifecycle_decay_days"]["provenance"] == "default"

    assert report["service_mode"]["value"] is False
    assert report["service_mode"]["provenance"] == "engine_override"


@pytest.mark.asyncio
async def test_env_var_set_to_the_code_default_still_reports_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Provenance follows the SOURCE, not the value.

    An operator who exports a variable has configured it, even when the chosen
    value coincides with the code default. Deriving provenance by comparing
    values reported ``default`` here and hid a real piece of run configuration.

    The same rule decides the engine branch, so it is asserted here too:
    ``storage_backend`` is computed by library-mode ``_build_settings`` (its
    rule collapses everything that is not redis to "inprocess"), and it reports
    ``engine_override`` even when the environment happens to hold the value the
    engine arrived at. Reporting ``env`` there would credit a variable the
    engine's rule can override outright.
    """
    default_rrf_k = Settings.from_env({}).rrf_k
    monkeypatch.setenv("ATAGIA_RRF_K", str(default_rrf_k))
    monkeypatch.setenv("ATAGIA_STORAGE_BACKEND", "inprocess")

    async with _engine(tmp_path / "value-equal-default.db") as engine:
        report = engine.effective_settings_report()["settings"]

    assert report["rrf_k"]["value"] == default_rrf_k
    assert report["rrf_k"]["provenance"] == "env"
    assert report["storage_backend"]["value"] == "inprocess"
    assert report["storage_backend"]["provenance"] == "engine_override"


@pytest.mark.asyncio
async def test_report_is_frozen_at_setup_and_ignores_later_env_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The report is an audit artifact of what ran.

    It is captured from the Settings handed to the runtime, so mutating the
    environment afterwards (benchmark harnesses in this repo do exactly that)
    cannot rewrite history.
    """
    monkeypatch.setenv("ATAGIA_RRF_K", "11")

    async with _engine(tmp_path / "frozen.db") as engine:
        before = engine.effective_settings_report()
        assert engine.runtime is not None
        assert engine.runtime.settings.rrf_k == 11

        monkeypatch.setenv("ATAGIA_RRF_K", "999")
        after = engine.effective_settings_report()

    assert before["settings"]["rrf_k"]["value"] == 11
    assert after["settings"]["rrf_k"]["value"] == 11
    assert after == before


def test_report_requires_a_run(tmp_path: Path) -> None:
    """Without setup there is no run to report on: fail fast instead of
    inventing a report from the current environment."""
    engine = _engine(tmp_path / "never-started.db")

    with pytest.raises(RuntimeNotInitializedError, match="call setup"):
        engine.effective_settings_report()


@pytest.mark.asyncio
async def test_resolved_policy_carries_manifest_provenance(
    tmp_path: Path,
) -> None:
    """The manifest layer CS-1.3 names is present: the retrieval policy each
    mode resolves to, tagged ``manifest`` and separate from the Settings block."""
    async with _engine(tmp_path / "policy.db") as engine:
        report = engine.effective_settings_report()
        assert engine.runtime is not None
        expected_modes = set(engine.runtime.manifests)

    policies = report["resolved_policy"]
    assert set(policies) == expected_modes
    assert expected_modes

    general = policies["general_qa"]
    for policy_field in (
        "context_budget_tokens",
        "transcript_budget_tokens",
        "privacy_ceiling",
        "retrieval_params",
    ):
        assert general[policy_field]["provenance"] == "manifest"
        assert general[policy_field]["redacted"] is False
    assert isinstance(general["context_budget_tokens"]["value"], int)
    assert general["retrieval_params"]["value"]["rerank_top_k"] > 0


@pytest.mark.asyncio
async def test_policy_fields_the_manifest_does_not_set_are_not_manifest_sourced(
    tmp_path: Path,
) -> None:
    """``manifest`` is a claim about the manifest FILE.

    The block used to tag every resolved field ``manifest`` unconditionally, so
    a hardcoded resolver constant, a schema default the file never declares, and
    a hash computed from the payload all claimed a source that did not produce
    them -- the same misattribution the settings block was already fixed for.
    """
    async with _engine(tmp_path / "policy-provenance.db") as engine:
        report = engine.effective_settings_report()

    general = report["resolved_policy"]["general_qa"]

    assert general["allow_intimacy_context"]["provenance"] == "default"
    assert general["cross_chat_allowed"]["provenance"] == "resolver"
    assert general["allowed_scopes"]["provenance"] == "resolver"
    assert general["allow_private_sensitivity"]["provenance"] == "resolver"
    assert general["prompt_hash"]["provenance"] == "computed"
    assert general["display_name"]["provenance"] == "manifest"

    for mode_policy in report["resolved_policy"].values():
        for field, entry in mode_policy.items():
            assert entry["provenance"] in {
                "manifest",
                "default",
                "resolver",
                "computed",
            }, field


def test_policy_report_requires_provenance_for_every_field() -> None:
    """A resolved-policy value with no provenance could only be tagged by
    guesswork, so the pair is rejected instead of silently mislabeled."""
    with pytest.raises(ValueError, match="does not cover"):
        ResolvedPolicyReport(
            values={"privacy_ceiling": 2, "cross_chat_allowed": True},
            provenance={"privacy_ceiling": "manifest"},
        )


@pytest.mark.asyncio
async def test_secrets_redacted_and_never_leak(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A set secret emits ``<redacted:set>`` and its value never appears in the
    serialized report; an empty secret emits ``<redacted:empty>``."""
    sentinel = "SENTINEL_SECRET_ABC123"
    monkeypatch.setenv("ATAGIA_ANTHROPIC_API_KEY", sentinel)

    async with _engine(tmp_path / "secret.db") as engine:
        report = engine.effective_settings_report()

    anthropic_entry = report["settings"]["anthropic_api_key"]
    assert anthropic_entry["redacted"] is True
    assert anthropic_entry["value"] == REDACTED_SET

    # admin_api_key is engine-forced to None, so it redacts to the empty sentinel
    # (presence still auditable) rather than exposing anything.
    admin_entry = report["settings"]["admin_api_key"]
    assert admin_entry["redacted"] is True
    assert admin_entry["value"] == REDACTED_EMPTY

    serialized = json.dumps(report)
    assert sentinel not in serialized
    assert REDACTED_SET in serialized


@pytest.mark.asyncio
async def test_url_userinfo_credentials_never_reach_the_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A redis-style URL carrying user:password userinfo must emit with the
    userinfo mechanically scrubbed — host/port stay auditable, the credential
    never survives serialization."""
    password_sentinel = "SUPERSECRET_PW_XYZ789"
    monkeypatch.setenv(
        "ATAGIA_REDIS_URL",
        f"redis://default:{password_sentinel}@cache.example.com:6379/0",
    )

    async with _engine(tmp_path / "redis.db") as engine:
        report = engine.effective_settings_report()

    redis_entry = report["settings"]["redis_url"]
    assert password_sentinel not in json.dumps(report)
    assert "<redacted-userinfo>@cache.example.com:6379" in redis_entry["value"]
    assert redis_entry["value"].startswith("redis://")


@pytest.mark.asyncio
async def test_url_query_string_credentials_never_reach_the_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Base-URL fields are not secret-shaped by name, yet a query string is a
    standard credential carrier (``?key=`` is the Google/Gemini REST form)."""
    query_sentinel = "SUPERSECRET_QUERY_TOKEN_42"
    monkeypatch.setenv(
        "ATAGIA_OPENAI_BASE_URL",
        f"https://gateway.example/v1?api_key={query_sentinel}&region=eu",
    )
    monkeypatch.setenv(
        "ATAGIA_ANTHROPIC_BASE_URL",
        f"https://generative.example/v1beta?key={query_sentinel}",
    )

    async with _engine(tmp_path / "query-url.db") as engine:
        report = engine.effective_settings_report()

    assert query_sentinel not in json.dumps(report)
    openai_value = report["settings"]["openai_base_url"]["value"]
    assert openai_value == (
        f"https://gateway.example/v1?api_key={REDACTED_SET}&region=eu"
    )
    assert (
        report["settings"]["anthropic_base_url"]["value"]
        == f"https://generative.example/v1beta?key={REDACTED_SET}"
    )


def test_unparseable_url_shaped_value_fails_closed() -> None:
    """A string that looks like a URL but cannot be parsed redacts entirely:
    we cannot prove it is credential-free.

    ``urlsplit`` is lazy, so the shape that raises decides WHERE it raises: the
    bracket error comes out of the split itself, a malformed port only out of the
    later ``.port`` access. The contract is the same for both, and it does not
    depend on the URL happening to carry userinfo -- reading the port only in the
    credential-bearing branch crashed the report on one shape and returned the
    other unscrubbed.
    """
    from atagia.core.effective_settings import REDACTED_SET, _scrub_url_credentials

    assert _scrub_url_credentials("redis://user:pw@[bad-ipv6:9999999") == (
        REDACTED_SET,
        True,
    )
    assert _scrub_url_credentials("redis://user:pw@host:bad/0") == (REDACTED_SET, True)
    assert _scrub_url_credentials("redis://host:bad/0") == (REDACTED_SET, True)
    assert _scrub_url_credentials("plain string, no url") == (
        "plain string, no url",
        False,
    )
    assert _scrub_url_credentials("https://example.com/path?q=1") == (
        "https://example.com/path?q=1",
        False,
    )
    # Non-credential parameters survive byte-for-byte alongside a redacted one.
    assert _scrub_url_credentials("https://example.com/p?a=1&access_token=zz&b=2") == (
        f"https://example.com/p?a=1&access_token={REDACTED_SET}&b=2",
        True,
    )


@pytest.mark.asyncio
async def test_malformed_url_setting_is_reported_not_crashed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A malformed URL-shaped setting must not take the run down.

    ``Settings.from_env`` accepts it, so the report is the first component to
    parse it; raising there aborted ``setup()`` for a value the engine itself was
    willing to run with. It fails closed instead, and the rejected FIELD is named
    by its own entry carrying ``redacted``.
    """
    monkeypatch.setenv("ATAGIA_REDIS_URL", "redis://default:pw@cache.example:bad/0")

    async with _engine(tmp_path / "malformed-url.db") as engine:
        report = engine.effective_settings_report()["settings"]

    assert report["redis_url"]["value"] == REDACTED_SET
    assert report["redis_url"]["redacted"] is True
    assert "pw" not in json.dumps(report["redis_url"])


@pytest.mark.asyncio
async def test_redacted_flag_marks_scrubbed_values_not_just_secret_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``redacted`` reports whether the emitted value was altered.

    A URL field is not secret-shaped by name, so without this the report showed a
    scrubbed value while claiming nothing had been redacted, and no field named
    itself as the one that lost content.
    """
    monkeypatch.setenv("ATAGIA_REDIS_URL", "redis://default:pw@cache.example:6379/0")
    monkeypatch.setenv("ATAGIA_OPENROUTER_SITE_URL", "https://site.example/atagia")

    async with _engine(tmp_path / "redacted-flag.db") as engine:
        report = engine.effective_settings_report()["settings"]

    assert report["redis_url"]["redacted"] is True
    assert report["openrouter_site_url"]["redacted"] is False
    assert report["openrouter_site_url"]["value"] == "https://site.example/atagia"


@pytest.mark.asyncio
async def test_from_env_fallback_fields_report_their_real_source_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fields whose defaults live inside from_env fallbacks (not dataclass
    defaults) must not be mistaken for configured values when their env var is
    unset -- and which layer they are attributed to still follows names.

    ``openrouter_site_url`` is untouched by library mode, so an unset variable
    leaves it at ``default``. ``redis_url`` IS in the engine's override
    allowlist, but library mode only writes it as ``self._redis_url or
    env_settings.redis_url``: with no constructor argument the engine hands back
    the value it was given, so an unset variable leaves it at ``default`` too.
    Tagging it ``engine_override`` here would credit the engine for a value the
    ``from_env`` fallback produced.
    """
    monkeypatch.delenv("ATAGIA_REDIS_URL", raising=False)
    monkeypatch.delenv("ATAGIA_OPENROUTER_SITE_URL", raising=False)

    async with _engine(tmp_path / "baseline.db") as engine:
        report = engine.effective_settings_report()["settings"]

    assert report["openrouter_site_url"]["provenance"] == "default"
    assert report["redis_url"]["value"] == Settings.from_env({}).redis_url
    assert report["redis_url"]["provenance"] == "default"


@pytest.mark.asyncio
async def test_allowlisted_field_reports_env_until_the_caller_supplies_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Membership in the override allowlist is not evidence the engine sourced
    the value.

    ``llm_chat_model`` is allowlisted, but library mode writes it as
    ``self._llm_chat_model or env_settings.llm_chat_model``. The SAME field is
    reported ``env`` when the constructor argument is absent (the engine handed
    the env value straight back) and ``engine_override`` when the caller passes
    it (the engine overruled the environment).
    """
    monkeypatch.setenv("ATAGIA_LLM_CHAT_MODEL", "env-provider/env-chat-model")

    async with Atagia(
        db_path=tmp_path / "chat-model-from-env.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    ) as engine:
        from_env = engine.effective_settings_report()["settings"]["llm_chat_model"]

    assert from_env["value"] == "env-provider/env-chat-model"
    assert from_env["provenance"] == "env"

    async with Atagia(
        db_path=tmp_path / "chat-model-from-caller.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
        llm_chat_model="caller-provider/caller-chat-model",
    ) as engine:
        from_caller = engine.effective_settings_report()["settings"]["llm_chat_model"]

    assert from_caller["value"] == "caller-provider/caller-chat-model"
    assert from_caller["provenance"] == "engine_override"


@pytest.mark.asyncio
async def test_engine_forced_fields_report_engine_override_without_caller_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Library mode pins a few fields on its own authority: they are the
    engine's whatever the caller passed and whatever the environment holds,
    because library mode is not a service and runs its own workers. The
    environment says the opposite for every one of them here."""
    monkeypatch.setenv("ATAGIA_WORKERS_ENABLED", "false")
    monkeypatch.setenv("ATAGIA_SERVICE_MODE", "true")
    monkeypatch.setenv("ATAGIA_ALLOW_INSECURE_HTTP", "false")
    monkeypatch.setenv("ATAGIA_ADMIN_API_KEY", "env-admin-key")

    async with _engine(tmp_path / "engine-forced.db") as engine:
        report = engine.effective_settings_report()["settings"]

    for field in _ENGINE_FORCED_SETTINGS_FIELDS:
        assert report[field]["provenance"] == "engine_override", field
    assert report["workers_enabled"]["value"] is True
    assert report["service_mode"]["value"] is False
    assert report["allow_insecure_http"]["value"] is True
    assert report["admin_api_key"]["value"] == REDACTED_EMPTY


@pytest.mark.asyncio
async def test_skip_flags_follow_env_and_report_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``skip_belief_revision`` / ``skip_compaction`` are env configuration.

    Their constructor parameters used to default to ``False`` instead of
    ``None``, so omitting them was indistinguishable from passing ``False``: the
    engine wrote its own default over ``ATAGIA_SKIP_*`` on every boot and the
    report called that an engine override. Both env variables are declared
    configuration, so they must reach the runtime and be reported as env.
    """
    monkeypatch.setenv("ATAGIA_SKIP_COMPACTION", "true")
    monkeypatch.setenv("ATAGIA_SKIP_BELIEF_REVISION", "true")

    async with _engine(tmp_path / "skip-from-env.db") as engine:
        report = engine.effective_settings_report()["settings"]
        assert engine.runtime is not None
        assert engine.runtime.settings.skip_compaction is True
        assert engine.runtime.settings.skip_belief_revision is True

    for field in ("skip_compaction", "skip_belief_revision"):
        assert report[field]["value"] is True, field
        assert report[field]["provenance"] == "env", field


@pytest.mark.asyncio
async def test_skip_flags_supplied_by_the_caller_still_win_over_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The sentinel keeps the caller in charge: an explicitly passed ``False``
    overrules the environment and is reported as the engine's, which is exactly
    what the old real-valued default could not express."""
    monkeypatch.setenv("ATAGIA_SKIP_COMPACTION", "true")
    monkeypatch.setenv("ATAGIA_SKIP_BELIEF_REVISION", "true")

    async with Atagia(
        db_path=tmp_path / "skip-from-caller.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
        skip_compaction=False,
        skip_belief_revision=False,
    ) as engine:
        report = engine.effective_settings_report()["settings"]
        assert engine.runtime is not None
        assert engine.runtime.settings.skip_compaction is False

    for field in ("skip_compaction", "skip_belief_revision"):
        assert report[field]["value"] is False, field
        assert report[field]["provenance"] == "engine_override", field


@pytest.mark.asyncio
async def test_sqlite_path_from_env_reaches_the_runtime_and_reports_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``ATAGIA_SQLITE_PATH`` must open the database a run actually uses.

    The constructor's ``db_path`` used to default to the literal ``"atagia.db"``,
    which the engine wrote over the environment on every boot -- and which
    disagreed with ``Settings.from_env``'s own ``./data/atagia.db``, so library
    mode and service mode opened different files with no configuration at all.
    """
    env_db_path = tmp_path / "from-env.db"
    monkeypatch.setenv("ATAGIA_SQLITE_PATH", str(env_db_path))

    async with Atagia(
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    ) as engine:
        report = engine.effective_settings_report()["settings"]
        assert engine.runtime is not None
        assert engine.runtime.settings.sqlite_path == str(env_db_path)

    assert report["sqlite_path"]["value"] == str(env_db_path)
    assert report["sqlite_path"]["provenance"] == "env"
    assert env_db_path.is_file()


def test_library_and_service_mode_share_one_sqlite_path_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With nothing configured, both modes resolve the same database file."""
    monkeypatch.delenv("ATAGIA_SQLITE_PATH", raising=False)
    engine = Atagia()

    assert engine._build_settings().sqlite_path == Settings.from_env({}).sqlite_path


@pytest.mark.asyncio
async def test_env_selected_redis_is_reported_as_env_not_an_engine_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``ATAGIA_STORAGE_BACKEND=redis`` is the ENVIRONMENT choosing redis.

    The engine only carries that choice through, so crediting itself made the
    report internally inconsistent: it claimed the engine picked the backend
    while reporting the URL that backend uses as env-sourced.
    """
    monkeypatch.setenv("ATAGIA_STORAGE_BACKEND", "redis")
    monkeypatch.setenv("ATAGIA_REDIS_URL", "redis://cache.example:6379/0")

    engine = Atagia(
        db_path=tmp_path / "env-redis.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )
    settings = engine._build_settings()

    assert settings.storage_backend == "redis"
    assert "storage_backend" not in engine._engine_override_fields
    assert "redis_url" not in engine._engine_override_fields


def test_caller_selected_redis_is_reported_as_an_engine_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A ``redis_url`` argument is the caller's choice, so the engine layer is
    the source of the backend it implies -- even with the env asking for
    inprocess."""
    monkeypatch.setenv("ATAGIA_STORAGE_BACKEND", "inprocess")

    engine = Atagia(
        db_path=tmp_path / "caller-redis.db",
        redis_url="redis://caller.example:6379/0",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
    )
    settings = engine._build_settings()

    assert settings.storage_backend == "redis"
    assert "storage_backend" in engine._engine_override_fields
    assert "redis_url" in engine._engine_override_fields


@pytest.mark.asyncio
async def test_engine_override_fields_are_forced_plus_caller_supplied(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The per-boot provenance set is exactly the engine-forced fields plus what
    this caller supplied, and it never escapes the structural allowlist."""
    monkeypatch.setenv("ATAGIA_EMBEDDING_BACKEND", "none")
    monkeypatch.setenv("ATAGIA_ANSWER_STANCE", "reactive")

    async with Atagia(
        db_path=tmp_path / "override-set.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
        answer_stance="proactive",
    ) as engine:
        override_fields = engine._engine_override_fields
        report = engine.effective_settings_report()["settings"]

    assert override_fields <= _ENGINE_SETTINGS_OVERRIDE_FIELDS
    assert _ENGINE_FORCED_SETTINGS_FIELDS <= override_fields
    assert override_fields == _ENGINE_FORCED_SETTINGS_FIELDS | {
        # Supplied by this caller.
        "sqlite_path",
        "openai_api_key",
        "llm_forced_global_model",
        "answer_stance",
        # Nothing asked for redis, so the collapse-to-inprocess floor is the
        # engine's own rule rather than a value the environment chose.
        "storage_backend",
    }

    # And the report agrees field by field with that set.
    for field in _ENGINE_SETTINGS_OVERRIDE_FIELDS:
        expected = "engine_override" if field in override_fields else "env"
        if field not in override_fields and not any(
            env_var in os.environ for env_var in SETTINGS_ENV_VARS[field]
        ):
            expected = "default"
        assert report[field]["provenance"] == expected, field


@pytest.mark.asyncio
async def test_supplied_argument_the_merge_ignores_reports_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Provenance mirrors the merge branch, not the mere presence of a keyword.

    An empty string loses the ``or`` merge, so ``embedding_backend`` still comes
    from the environment; an empty mapping contributes nothing to the
    component-model merge, so that stays env-derived too. A falsy value the
    merge DOES take (a sentinel-guarded ``0``) is the engine's.
    """
    monkeypatch.setenv("ATAGIA_EMBEDDING_BACKEND", "none")
    monkeypatch.setenv("ATAGIA_LLM_MODEL__EXTRACTOR", "openai/env-extractor")

    async with Atagia(
        db_path=tmp_path / "ignored-argument.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
        embedding_backend="",
        llm_component_models={},
        llm_structured_output_retry_attempts=0,
    ) as engine:
        report = engine.effective_settings_report()["settings"]

    assert report["embedding_backend"]["value"] == "none"
    assert report["embedding_backend"]["provenance"] == "env"
    assert report["llm_component_models"]["value"] == (
        Settings.from_env().llm_component_models
    )
    assert (
        report["llm_component_models"]["value"]["extractor"] == "openai/env-extractor"
    )
    assert report["llm_component_models"]["provenance"] == "env"
    assert report["llm_structured_output_retry_attempts"]["value"] == 0
    assert report["llm_structured_output_retry_attempts"]["provenance"] == (
        "engine_override"
    )


@pytest.mark.asyncio
async def test_phase_model_override_makes_component_models_engine_sourced(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``llm_component_models`` is a merge, not a fallback: a caller-supplied
    phase model filters env-configured components out of it, so the engine is
    the source even though the caller passed no component mapping."""
    monkeypatch.setenv("ATAGIA_LLM_MODEL__EXTRACTOR", "openai/env-extractor")

    async with Atagia(
        db_path=tmp_path / "phase-filtered.db",
        openai_api_key="test-openai-key",
        llm_forced_global_model="openai/test-model",
        llm_ingest_model="openai/caller-ingest",
    ) as engine:
        report = engine.effective_settings_report()["settings"]

    assert report["llm_component_models"]["provenance"] == "engine_override"
    # The env-configured extractor is an ingest component, so the caller's
    # ingest model filtered it out: the mapping is no longer the env-derived one.
    assert Settings.from_env().llm_component_models["extractor"] == (
        "openai/env-extractor"
    )
    assert "extractor" not in report["llm_component_models"]["value"]
    assert report["llm_ingest_model"]["provenance"] == "engine_override"
