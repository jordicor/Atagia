"""Path-safe transport encoding for externally supplied identifiers."""

from __future__ import annotations

import base64
import binascii
import re

TRANSPORT_ID_PREFIX = "__atagia_b64_"
_SAFE_PATH_ID = re.compile(r"^[A-Za-z0-9_:-][A-Za-z0-9_.:-]*$")
_URLSAFE_BASE64 = re.compile(r"^[A-Za-z0-9_-]*$")


def encode_path_id(value: str) -> str:
    """Return a route-segment-safe, reversible representation of an id."""
    if _is_plain_safe(value):
        return value
    encoded = base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii").rstrip("=")
    return f"{TRANSPORT_ID_PREFIX}{encoded}"


def decode_path_id(value: str) -> str:
    """Decode a route id produced by encode_path_id, leaving plain ids unchanged."""
    if not value.startswith(TRANSPORT_ID_PREFIX):
        return value
    encoded = value[len(TRANSPORT_ID_PREFIX) :]
    if _URLSAFE_BASE64.fullmatch(encoded) is None:
        raise ValueError("Invalid Atagia transport id")
    padding = "=" * (-len(encoded) % 4)
    try:
        return base64.urlsafe_b64decode(f"{encoded}{padding}".encode("ascii")).decode(
            "utf-8"
        )
    except (binascii.Error, UnicodeDecodeError) as exc:
        raise ValueError("Invalid Atagia transport id") from exc


def _is_plain_safe(value: str) -> bool:
    return (
        value not in {".", ".."}
        and not value.startswith(TRANSPORT_ID_PREFIX)
        and _SAFE_PATH_ID.fullmatch(value) is not None
    )
