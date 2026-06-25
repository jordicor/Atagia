"""Environment variable parsing helpers."""

from __future__ import annotations

import os

_TRUE_ENV_TOKENS = {"1", "true", "yes", "on"}
_FALSE_ENV_TOKENS = {"0", "false", "no", "off"}


def env_bool(name: str, default: bool) -> bool:
    """Parse a boolean environment variable with fail-fast invalid values."""
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if not normalized:
        return default
    if normalized in _TRUE_ENV_TOKENS:
        return True
    if normalized in _FALSE_ENV_TOKENS:
        return False
    raise ValueError(f"{name} must be a boolean value")


def env_bool_optional(name: str, env: dict[str, str] | None = None) -> bool | None:
    """Parse an optional boolean env var.

    Returns None when the variable is unset or empty, so callers can layer a
    per-key override on top of a global default. Invalid values fail fast.
    """
    source = env if env is not None else os.environ
    value = source.get(name)
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized in _TRUE_ENV_TOKENS:
        return True
    if normalized in _FALSE_ENV_TOKENS:
        return False
    raise ValueError(f"{name} must be a boolean value")
