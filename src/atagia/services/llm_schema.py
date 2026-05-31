"""Utilities for simplifying schemas before they are sent to LLM providers."""

from __future__ import annotations

import copy
from typing import Any


def strip_json_schema_nullability(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a model-facing schema that represents unknown optionals by omission.

    Pydantic uses JSON Schema null unions to express ``None`` defaults. Those
    unions are useful for internal validation, but they make native structured
    output grammars larger and encourage models to emit explicit ``null``. For
    LLM-facing schemas, optional fields should be omitted when unknown instead.
    """

    cleaned = _strip_nullability(copy.deepcopy(schema))
    return cleaned if isinstance(cleaned, dict) else {}


def has_json_schema_nullability(node: Any) -> bool:
    """Return whether a schema tree explicitly allows JSON null values."""

    if isinstance(node, list):
        return any(has_json_schema_nullability(item) for item in node)
    if not isinstance(node, dict):
        return False

    if node.get("nullable") is True:
        return True

    node_type = node.get("type")
    if isinstance(node_type, list) and "null" in node_type:
        return True
    if node_type == "null":
        return True

    for key in ("anyOf", "oneOf"):
        variants = node.get(key)
        if isinstance(variants, list) and any(
            _is_null_schema(variant) for variant in variants
        ):
            return True

    return any(has_json_schema_nullability(value) for value in node.values())


def _strip_nullability(node: Any) -> Any:
    if isinstance(node, list):
        return [_strip_nullability(item) for item in node]
    if not isinstance(node, dict):
        return node

    cleaned: dict[str, Any] = {}
    for key, value in node.items():
        if key == "nullable":
            continue
        if key == "default" and value is None:
            continue
        if key == "type" and isinstance(value, list):
            non_null_types = [item for item in value if item != "null"]
            if not non_null_types:
                continue
            cleaned[key] = (
                non_null_types[0] if len(non_null_types) == 1 else non_null_types
            )
            continue
        if key in {"anyOf", "oneOf"} and isinstance(value, list):
            variants = [
                _strip_nullability(variant)
                for variant in value
                if not _is_null_schema(variant)
            ]
            if len(variants) == 1 and isinstance(variants[0], dict):
                for child_key, child_value in variants[0].items():
                    cleaned.setdefault(child_key, child_value)
            elif variants:
                cleaned[key] = variants
            continue
        cleaned[key] = _strip_nullability(value)

    if cleaned.get("type") == "null":
        cleaned.pop("type")
    return cleaned


def _is_null_schema(node: Any) -> bool:
    return isinstance(node, dict) and node.get("type") == "null"
