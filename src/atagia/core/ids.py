"""Prefixed identifier generation."""

from __future__ import annotations

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


def new_belief_id() -> str:
    return generate_prefixed_id("blf")

