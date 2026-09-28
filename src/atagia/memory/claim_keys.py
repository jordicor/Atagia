"""Mechanical syntax contract for internal belief claim keys."""

from __future__ import annotations

import re


_CLAIM_KEY_PATTERN = re.compile(
    r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*(?:\.[a-z][a-z0-9]*(?:_[a-z0-9]+)*)*",
    re.ASCII,
)


def validate_claim_key(value: str) -> str:
    """Return an unchanged canonical key or reject invalid identifier syntax.

    This checks spelling format only. The producer must choose English concepts;
    ASCII syntax cannot determine the language or semantic equivalence of a key.
    """
    if not isinstance(value, str) or _CLAIM_KEY_PATTERN.fullmatch(value) is None:
        raise ValueError(
            "claim_key must use lowercase ASCII letters, digits and underscores "
            "within dot-separated segments, starting each segment with a letter"
        )
    return value
