"""Tests for canonical JSON helpers."""

from __future__ import annotations

from atagia.core.canonical import canonical_json_bytes, canonical_json_hash


def test_canonical_json_bytes_are_stable_across_key_order() -> None:
    left = {"b": [2, 1], "a": {"z": True, "m": None}}
    right = {"a": {"m": None, "z": True}, "b": [2, 1]}

    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert canonical_json_hash(left) == canonical_json_hash(right)


def test_canonical_json_bytes_use_compact_utf8_json() -> None:
    payload = {"text": "cafe", "count": 2}

    assert canonical_json_bytes(payload) == b'{"count":2,"text":"cafe"}'
