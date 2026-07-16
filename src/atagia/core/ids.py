"""Prefixed identifier generation."""

from __future__ import annotations

import hashlib
import re
from secrets import token_hex

_PREFIX_PATTERN = re.compile(r"^[a-z][a-z0-9]*$")


def generate_prefixed_id(prefix: str) -> str:
    """Return an opaque identifier with the given prefix."""
    normalized = prefix.removesuffix("_").strip().lower()
    if not _PREFIX_PATTERN.fullmatch(normalized):
        raise ValueError(f"Invalid identifier prefix: {prefix!r}")
    return f"{normalized}_{token_hex(10)}"


def new_memory_id() -> str:
    return generate_prefixed_id("mem")


def new_retrieval_id() -> str:
    return generate_prefixed_id("ret")


def new_job_id() -> str:
    return generate_prefixed_id("job")


def derive_child_job_id(
    parent_job_id: str,
    job_type: str,
    logical_key: str,
) -> str:
    """Return a stable opaque identity for one logical child-job intent."""

    digest = hashlib.sha256(
        "\x1f".join((parent_job_id, job_type, logical_key)).encode("utf-8")
    ).hexdigest()[:20]
    return f"job_{digest}"


def new_belief_id() -> str:
    return generate_prefixed_id("blf")
