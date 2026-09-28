"""Version 1 of the local diagnostic capture format.

A capture is one directory containing manifest.json, events.jsonl and
content-addressed blobs/<sha256>. Events are ordered by seq starting at 1.
Every blob reference names the exact bytes needed to reconstruct the event;
IDs and hashes alone are not accepted as source evidence. A capture is
reproducible only when the manifest is complete and every hash validates.

The format is deliberately independent of the private analysis program.
Unknown versions must be rejected rather than interpreted as this version.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

SCHEMA_VERSION = 1
EVENT_KINDS = frozenset({"operation_start", "operation_end", "provider_attempt", "provider_payload", "provider_raw", "no_call"})
CAPTURE_STATUSES = frozenset({"incomplete", "complete", "failed"})
ATTEMPT_STATUSES = frozenset({"success", "failure", "cancelled", "partial", "blocked"})


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a format record without locale or whitespace ambiguity."""
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
