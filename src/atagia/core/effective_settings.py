"""Effective-settings report: what a run actually executed with, per field.

This produces an auditable snapshot proving the effective configuration a run
used. It has two blocks:

* ``settings`` -- every ``Settings`` field with its effective value and a
  provenance tag.
* ``resolved_policy`` -- the retrieval policy resolved from the mode manifests
  for that run, per assistant mode, each field tagged with the layer that
  actually produced it (``manifest`` / ``default`` / ``resolver`` /
  ``computed``, see ``atagia.memory.policy_manifest``). A field the manifest
  file never declares is not ``manifest``.

Settings provenance is derived from the real source, never from value equality:

* ``engine_override`` -- ``_build_settings`` is what decided the value, either
  because library mode pins the field outright or because the caller handed the
  engine the value through the ``Atagia`` constructor. Membership in the set the
  engine computes for that boot is the whole test: an override that happens to
  land on the same value the environment held still came from the engine, and
  reporting it as ``env`` would credit the wrong layer. Symmetrically, a field
  the engine merely copied back from the environment because the caller supplied
  nothing is NOT in that set, so it reports its real source (``env`` /
  ``default``).
* ``env`` -- one of the environment variables the field is read from was
  present in the environment at read time (declared in ``SETTINGS_ENV_VARS``),
  so the value came from the environment even when it equals the code default.
* ``default`` -- no environment variable feeding the field was set.

Secret-shaped fields (matched mechanically by field NAME) never expose their
value: they emit ``<redacted:set>`` / ``<redacted:empty>`` so presence stays
auditable without leaking the credential itself. Strings are additionally
scrubbed for URL-embedded credentials (userinfo and credential-shaped query
parameters) because a base URL field is not itself secret-shaped.

Each entry's ``redacted`` flag reports whether the emitted value was altered by
either mechanism, so the field whose value was rejected or rewritten is named by
its own entry instead of being indistinguishable from a value reported verbatim.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from atagia.core.config import Settings
from atagia.core.settings_env_vars import SETTINGS_ENV_VARS

REDACTED_SET = "<redacted:set>"
REDACTED_EMPTY = "<redacted:empty>"
REDACTED_USERINFO = "<redacted-userinfo>"

# Settings-block provenance. The resolved-policy block has its own vocabulary,
# owned by `atagia.memory.policy_manifest` because only the resolver knows which
# layer produced each policy field.
PROVENANCE_DEFAULT = "default"
PROVENANCE_ENV = "env"
PROVENANCE_ENGINE_OVERRIDE = "engine_override"

# URL query parameters carry credentials under naming conventions that field
# names never use (``?key=`` is the standard Google/Gemini REST form). These
# extend the field-name markers for query keys only; matching is mechanical on
# a machine-defined grammar, never on natural language.
_CREDENTIAL_QUERY_PARAMS = frozenset({"key", "apikey", "auth", "sig", "signature"})


def is_secret_field(field_name: str) -> bool:
    """Mechanical, name-only secret detection.

    Matches the credential markers ``api_key`` / ``secret`` / ``password`` as
    substrings and an auth ``token`` only as a whole field name or ``*_token``
    suffix. The ``*_token`` restriction is deliberate: this codebase names LLM
    token-COUNT settings with segments like ``token_threshold``, ``token_lag``,
    and the ``*_tokens`` plural, none of which are credentials and none of which
    may be redacted (redacting them would defeat the report's purpose).
    """
    if "api_key" in field_name or "secret" in field_name or "password" in field_name:
        return True
    return field_name == "token" or field_name.endswith("_token")


def _is_credential_query_param(name: str) -> bool:
    """Whether a URL query parameter name looks like it carries a credential."""
    normalized = name.strip().lower().replace("-", "_")
    return is_secret_field(normalized) or normalized in _CREDENTIAL_QUERY_PARAMS


def _redact(value: Any) -> str:
    is_set = value is not None and value != ""
    return REDACTED_SET if is_set else REDACTED_EMPTY


def _scrub_url_query(component: str) -> str:
    """Redact credential-shaped ``key=value`` pairs in a query or fragment.

    Splits on the URL grammar only and rewrites nothing else, so non-credential
    parameters survive byte-for-byte.
    """
    if "=" not in component:
        return component
    parts = component.split("&")
    scrubbed: list[str] = []
    for part in parts:
        name, separator, value = part.partition("=")
        if separator and _is_credential_query_param(name):
            scrubbed.append(f"{name}={REDACTED_SET if value else REDACTED_EMPTY}")
        else:
            scrubbed.append(part)
    return "&".join(scrubbed)


def _scrub_url_credentials(value: str) -> tuple[str, bool]:
    """Mechanically strip credentials embedded in a URL.

    Returns the emitted value and whether anything was redacted.

    Two carriers are covered: ``scheme://user:password@host`` userinfo and
    credential-shaped query/fragment parameters such as ``?api_key=`` or
    ``?key=``. Either would leak into the persisted manifest even though the
    field name is not secret-shaped. Structural URL parsing only. If a string
    LOOKS like a URL (contains ``://``) but cannot be parsed, the whole value is
    redacted (fail closed: we cannot prove it is credential-free).

    Every component this function reads is parsed inside the guard, not just the
    ``urlsplit`` call: ``urlsplit`` is lazy, so a malformed port or host raises
    from the ATTRIBUTE access. Reading them outside meant a malformed port
    crashed the report in the credential-bearing branch and slipped through
    unscrubbed in the branch that never touched the port -- the fail-closed
    contract holding for one URL shape and not the other.
    """
    if "://" not in value:
        return value, False
    try:
        parts = urlsplit(value)
        username = parts.username
        password = parts.password
        hostname = parts.hostname or ""
        port = parts.port
        raw_query = parts.query
        raw_fragment = parts.fragment
        scheme = parts.scheme
        path = parts.path
        netloc = parts.netloc
    except ValueError:
        return REDACTED_SET, True
    query = _scrub_url_query(raw_query)
    fragment = _scrub_url_query(raw_fragment)
    if username is None and password is None:
        if query == raw_query and fragment == raw_fragment:
            return value, False
    else:
        netloc = f"{REDACTED_USERINFO}@{hostname}"
        if port is not None:
            netloc = f"{netloc}:{port}"
    return urlunsplit((scheme, netloc, path, query, fragment)), True


def _json_safe(value: Any) -> tuple[Any, bool]:
    """Normalize a value to JSON-native containers (tuples -> lists), scrubbing
    URL-embedded credentials from every string on the way.

    Returns the normalized value and whether any string inside it was redacted.
    """
    if isinstance(value, (tuple, list)):
        items = [_json_safe(item) for item in value]
        return [item for item, _ in items], any(redacted for _, redacted in items)
    if isinstance(value, dict):
        entries = {key: _json_safe(item) for key, item in value.items()}
        return (
            {key: item for key, (item, _) in entries.items()},
            any(redacted for _, redacted in entries.values()),
        )
    if isinstance(value, str):
        return _scrub_url_credentials(value)
    return value, False


def _entry(name: str, value: Any, provenance: str) -> dict[str, Any]:
    if is_secret_field(name):
        return {"value": _redact(value), "provenance": provenance, "redacted": True}
    safe_value, redacted = _json_safe(value)
    return {"value": safe_value, "provenance": provenance, "redacted": redacted}


@dataclasses.dataclass(frozen=True, slots=True)
class ResolvedPolicyReport:
    """One mode's resolved retrieval policy with its per-field provenance.

    The two mappings are supplied together, and their key sets must match: a
    value with no provenance would have to be tagged by guesswork, which is the
    misattribution this report exists to prevent.
    """

    values: Mapping[str, Any]
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        if set(self.values) != set(self.provenance):
            missing = sorted(set(self.values) - set(self.provenance))
            extra = sorted(set(self.provenance) - set(self.values))
            raise ValueError(
                "resolved policy provenance does not cover the resolved policy "
                f"fields (missing={missing}, unknown={extra})"
            )


def build_effective_settings_report(
    *,
    effective_settings: Settings,
    present_env_names: frozenset[str],
    engine_override_fields: frozenset[str],
    resolved_policies: Mapping[str, ResolvedPolicyReport],
) -> dict[str, Any]:
    """Build the effective-configuration report with provenance + redaction.

    ``effective_settings`` is the final runtime configuration,
    ``present_env_names`` is the set of environment variable names present when
    the env base was read, ``engine_override_fields`` is the set of fields whose
    value the engine layer sourced on this boot (not merely wrote back from the
    env base), and ``resolved_policies`` maps an assistant mode id to the
    retrieval policy resolved from its manifest for this run, each field carrying
    the layer that produced it.

    The env-derived base itself is deliberately not an input: provenance is
    decided by name membership on both branches, so comparing values against it
    could only reintroduce the misattribution this report exists to avoid.

    Manifest values are the run-level resolution (no workspace, conversation, or
    operational override, which are per-request); the composed per-turn budget
    is additionally capped by the context envelope at request time.
    """
    settings_report: dict[str, Any] = {}
    for field in dataclasses.fields(Settings):
        name = field.name
        final_value = getattr(effective_settings, name)
        env_var_names = SETTINGS_ENV_VARS[name]

        # Membership decides this, never a value comparison. The engine sourced
        # every field in `engine_override_fields`, so it is the source even when
        # the value it produced coincides with the env-derived one.
        if name in engine_override_fields:
            provenance = PROVENANCE_ENGINE_OVERRIDE
        elif any(env_var in present_env_names for env_var in env_var_names):
            provenance = PROVENANCE_ENV
        else:
            provenance = PROVENANCE_DEFAULT

        settings_report[name] = _entry(name, final_value, provenance)

    policy_report: dict[str, Any] = {
        mode_id: {
            key: _entry(key, value, policy.provenance[key])
            for key, value in sorted(policy.values.items())
        }
        for mode_id, policy in sorted(resolved_policies.items())
    }
    return {"settings": settings_report, "resolved_policy": policy_report}
