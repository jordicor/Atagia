"""Canonical JSON helpers shared by config-like loaders."""

from __future__ import annotations

import hashlib
from typing import Any

from atagia.core import json_utils


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    """Return stable UTF-8 JSON bytes for hashing and cache keys."""
    return json_utils.dumps_bytes(
        payload,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_json_hash(payload: dict[str, Any]) -> str:
    """Return a SHA-256 hash over canonical JSON bytes."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
