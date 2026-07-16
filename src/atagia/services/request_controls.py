"""Canonical validation for request-scoped memory and authority controls."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


_TRUE_BOOLEAN_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_BOOLEAN_VALUES = frozenset({"0", "false", "no", "off"})

_MEMORY_SCOPE_FIELDS: dict[str, tuple[str, ...]] = {
    "incognito": ("incognito", "atagia_incognito"),
    "cross_chat_memory": (
        "cross_chat_memory",
        "atagia_cross_chat_memory",
    ),
}
_MEMORY_SCOPE_HEADERS: dict[str, str] = {
    "incognito": "x-atagia-incognito",
    "cross_chat_memory": "x-atagia-cross-chat-memory",
}

_REMOTE_AUTHORITY_FIELDS = frozenset(
    {
        "privacy_enforcement",
        "authenticated_user_privilege_level",
        "authenticated_user_is_atagia_master",
        "authenticated_privilege_level",
        "authenticated_atagia_master",
        "is_atagia_master",
        "atagia_master",
        "atagia_privacy_enforcement",
        "atagia_authenticated_user_privilege_level",
        "atagia_authenticated_user_is_atagia_master",
        "atagia_authenticated_privilege_level",
        "atagia_authenticated_atagia_master",
    }
)
_REMOTE_AUTHORITY_HEADERS = frozenset(
    {
        "x-atagia-privacy-enforcement",
        "x-atagia-authenticated-user-privilege-level",
        "x-atagia-authenticated-user-is-atagia-master",
        "x-atagia-authenticated-privilege-level",
        "x-atagia-authenticated-atagia-master",
    }
)


@dataclass(frozen=True, slots=True)
class ResolvedMemoryScopeControls:
    """Consistent effective memory-scope controls for one request."""

    incognito: bool | None
    cross_chat_memory: bool


@dataclass(frozen=True, slots=True)
class _BooleanClaim:
    source: str
    value: Any


def resolve_memory_scope_controls(
    *,
    typed_fields: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    headers: Mapping[str, Any] | None = None,
    default_cross_chat_memory: bool = True,
) -> ResolvedMemoryScopeControls:
    """Collect and resolve all accepted incognito and cross-chat claims.

    A setting may be repeated across typed fields, metadata aliases, and
    accepted headers only when every explicit boolean agrees. ``None`` is
    treated as an omitted optional value. Once each setting is internally
    consistent, incognito narrows cross-chat access but can never broaden it.
    """

    typed_fields = typed_fields or {}
    metadata = metadata or {}
    resolved: dict[str, bool | None] = {}

    for setting, aliases in _MEMORY_SCOPE_FIELDS.items():
        claims: list[_BooleanClaim] = []
        if setting in typed_fields and typed_fields[setting] is not None:
            claims.append(
                _BooleanClaim(
                    source=f"typed.{setting}",
                    value=typed_fields[setting],
                )
            )
        for alias in aliases:
            if alias in metadata and metadata[alias] is not None:
                claims.append(
                    _BooleanClaim(
                        source=f"metadata.{alias}",
                        value=metadata[alias],
                    )
                )
        header_name = _MEMORY_SCOPE_HEADERS[setting]
        for header_value in _header_values(headers, header_name):
            claims.append(
                _BooleanClaim(
                    source=f"header.{_display_header_name(header_name)}",
                    value=header_value,
                )
            )
        resolved[setting] = _resolve_boolean_claims(setting, claims)

    incognito = resolved["incognito"]
    cross_chat_memory = resolved["cross_chat_memory"]
    return ResolvedMemoryScopeControls(
        incognito=incognito,
        cross_chat_memory=(
            False
            if incognito is True
            else default_cross_chat_memory
            if cross_chat_memory is None
            else cross_chat_memory
        ),
    )


def reject_remote_authority_claims(
    *,
    metadata: Mapping[str, Any] | None = None,
    extra_fields: Mapping[str, Any] | None = None,
    headers: Mapping[str, Any] | None = None,
) -> None:
    """Reject ordinary HTTP attempts to supply server authority."""

    for source_name, values in (
        ("metadata", metadata or {}),
        ("body", extra_fields or {}),
    ):
        for key in values:
            normalized_key = str(key).strip().lower()
            if normalized_key in _REMOTE_AUTHORITY_FIELDS:
                raise ValueError(
                    f"Remote authority claim {source_name}.{key} is not allowed"
                )

    for key in headers or {}:
        normalized_key = str(key).strip().lower()
        if normalized_key in _REMOTE_AUTHORITY_HEADERS:
            raise ValueError(f"Remote authority claim header.{key} is not allowed")


def _resolve_boolean_claims(
    setting: str,
    claims: list[_BooleanClaim],
) -> bool | None:
    if not claims:
        return None
    parsed = [(_parse_boolean_claim(setting, claim), claim) for claim in claims]
    values = {value for value, _claim in parsed}
    if len(values) != 1:
        raise ValueError(f"Conflicting {setting} values across request sources")
    return parsed[0][0]


def _header_values(
    headers: Mapping[str, Any] | None,
    normalized_name: str,
) -> list[Any]:
    """Return every explicit instance of one case-insensitive header."""

    if headers is None:
        return []
    getlist = getattr(headers, "getlist", None)
    if callable(getlist):
        return [value for value in getlist(normalized_name) if value is not None]

    values: list[Any] = []
    for key, value in headers.items():
        if str(key).strip().lower() != normalized_name or value is None:
            continue
        if isinstance(value, (list, tuple)):
            values.extend(item for item in value if item is not None)
        else:
            values.append(value)
    return values


def _parse_boolean_claim(setting: str, claim: _BooleanClaim) -> bool:
    if isinstance(claim.value, bool):
        return claim.value
    if isinstance(claim.value, str):
        normalized = claim.value.strip().lower()
        if normalized in _TRUE_BOOLEAN_VALUES:
            return True
        if normalized in _FALSE_BOOLEAN_VALUES:
            return False
    raise ValueError(f"Invalid boolean for {setting} from {claim.source}")


def _display_header_name(normalized_name: str) -> str:
    return "-".join(part.capitalize() for part in normalized_name.split("-"))
