"""JSON artifact write helpers for benchmark tooling."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def write_json_atomic(path: str | Path, payload: Any) -> Path:
    """Write a JSON artifact through a same-directory atomic replacement."""
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        temp_path.replace(destination)
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise
    return destination
