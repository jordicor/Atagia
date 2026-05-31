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
