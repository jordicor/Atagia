"""Fast JSON helpers with stdlib-compatible fallbacks."""

from __future__ import annotations

import json as _json
from typing import Any

try:  # pragma: no cover - exercised when the optional speedup is installed.
    import orjson as _orjson
except ImportError:  # pragma: no cover - local dev may not have the extra yet.
    _orjson = None


JSONDecodeError = _json.JSONDecodeError


def loads(value: str | bytes | bytearray) -> Any:
    """Deserialize JSON, preferring orjson when available."""
    if _orjson is not None:
        try:
            return _orjson.loads(value)
        except _orjson.JSONDecodeError as exc:
            raise JSONDecodeError(str(exc), _coerce_text(value), getattr(exc, "pos", 0)) from exc
    return _json.loads(value)


def dumps(
    value: Any,
    *,
    ensure_ascii: bool = False,
    sort_keys: bool = False,
    separators: tuple[str, str] | None = None,
    indent: int | None = None,
) -> str:
    """Serialize JSON, using orjson for common UTF-8 output."""
    if _orjson is not None and not ensure_ascii and indent in (None, 2):
        option = _orjson.OPT_NON_STR_KEYS
        if sort_keys:
            option |= _orjson.OPT_SORT_KEYS
        if indent == 2:
            option |= _orjson.OPT_INDENT_2
        try:
            return _orjson.dumps(value, option=option).decode("utf-8")
        except TypeError:
            pass
    return _json.dumps(
        value,
        ensure_ascii=ensure_ascii,
        sort_keys=sort_keys,
        separators=separators,
        indent=indent,
    )


def dumps_bytes(
    value: Any,
    *,
    sort_keys: bool = False,
    separators: tuple[str, str] | None = None,
    indent: int | None = None,
) -> bytes:
    """Serialize JSON directly to UTF-8 bytes."""
    if _orjson is not None and indent in (None, 2):
        option = _orjson.OPT_NON_STR_KEYS
        if sort_keys:
            option |= _orjson.OPT_SORT_KEYS
        if indent == 2:
            option |= _orjson.OPT_INDENT_2
        try:
            return _orjson.dumps(value, option=option)
        except TypeError:
            pass
    return _json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=sort_keys,
        separators=separators,
        indent=indent,
    ).encode("utf-8")


def _coerce_text(value: str | bytes | bytearray) -> str:
    if isinstance(value, str):
        return value
    return bytes(value).decode("utf-8", errors="replace")
