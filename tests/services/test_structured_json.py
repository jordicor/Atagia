"""Tests for structured JSON helpers, including the compact schema spec."""

from __future__ import annotations

from pydantic import BaseModel, TypeAdapter

from atagia.models.schemas_memory import ExtractionResult, QueryIntelligenceResult
from atagia.services.structured_json import (
    COMPACT_SCHEMA_SPEC_MAX_CHARS,
    render_compact_schema_spec,
)


def _extraction_schema() -> dict:
    return TypeAdapter(ExtractionResult).json_schema()


def _query_intelligence_schema() -> dict:
    return TypeAdapter(QueryIntelligenceResult).json_schema()


def test_render_compact_schema_spec_is_deterministic_for_real_schemas() -> None:
    for schema in (_extraction_schema(), _query_intelligence_schema()):
        first = render_compact_schema_spec(schema)
        second = render_compact_schema_spec(schema)
        assert first == second
        assert first  # non-empty


def test_render_compact_schema_spec_honors_size_cap() -> None:
    # QueryIntelligenceResult fits under the cap when fully expanded.
    query_spec = render_compact_schema_spec(_query_intelligence_schema())
    assert len(query_spec) <= COMPACT_SCHEMA_SPEC_MAX_CHARS

    # ExtractionResult is large; the full expansion exceeds the cap, so the
    # renderer degrades to top-level structure + each element's fields once.
    extraction_schema = _extraction_schema()
    extraction_spec = render_compact_schema_spec(extraction_schema)
    # The degraded spec references element types at the top level and defines
    # each one once, instead of recursively expanding nested objects.
    assert "-> ExtractedEvidence" in extraction_spec
    assert "\nExtractedEvidence:" in extraction_spec
    # Far smaller than the raw JSON schema it replaces in the prompt.
    import json

    raw_len = len(json.dumps(extraction_schema, separators=(",", ":")))
    assert len(extraction_spec) < raw_len


def test_render_compact_schema_spec_includes_enum_values() -> None:
    spec = render_compact_schema_spec(_query_intelligence_schema())
    # NeedTrigger enum values are rendered inline.
    assert "enum:" in spec
    assert "ambiguity" in spec
    assert "under_specified_request" in spec
    # RuntimeAnchor anchor_type enum values too.
    assert "proper_name" in spec


def test_render_compact_schema_spec_marks_required_and_optional() -> None:
    spec = render_compact_schema_spec(_query_intelligence_schema())
    assert "[required]" in spec
    assert "[optional]" in spec
    # sub_queries is required on QueryIntelligenceResult.
    assert "sub_queries (array of string) [required]" in spec


def test_render_compact_schema_spec_renders_array_element_fields_once() -> None:
    spec = render_compact_schema_spec(_query_intelligence_schema())
    # The DetectedNeed element fields appear once under `needs`.
    assert "needs (array of object) [required]" in spec or "needs (array of object) [optional]" in spec
    assert spec.count("need_type (string)") == 1
    assert "reasoning (string) [required]" in spec


def test_render_compact_schema_spec_emits_only_schema_derived_tokens() -> None:
    schema = _query_intelligence_schema()
    spec = render_compact_schema_spec(schema)

    # Allowed structural tokens used by the renderer itself (the alpha-token
    # regex strips punctuation such as the ``enum:`` colon and bracket markers).
    structural_tokens = {
        "enum",
        "array",
        "of",
        "object",
        "string",
        "number",
        "integer",
        "boolean",
        "any",
        "required",
        "optional",
        "more",
    }

    # Collect schema-derived tokens: property names, $defs names, enum values,
    # and primitive type names.
    allowed: set[str] = set()

    def _collect(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "properties" and isinstance(value, dict):
                    allowed.update(value.keys())
                if key == "enum" and isinstance(value, list):
                    allowed.update(str(item) for item in value)
                if key == "type" and isinstance(value, str):
                    allowed.add(value)
                if key == "$ref" and isinstance(value, str):
                    allowed.add(value.rsplit("/", 1)[-1])
                _collect(value)
        elif isinstance(node, list):
            for item in node:
                _collect(item)

    _collect(schema)
    allowed.update(schema.get("$defs", {}).keys())

    # Every alphabetic token in the spec must be either a structural token or a
    # schema-derived token. No request/user data should leak in.
    import re

    for raw_token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", spec):
        assert (
            raw_token in structural_tokens
            or raw_token in allowed
        ), f"unexpected token in spec: {raw_token!r}"


def test_render_compact_schema_spec_handles_simple_model() -> None:
    class Simple(BaseModel):
        label: str
        score: int

    spec = render_compact_schema_spec(Simple.model_json_schema())
    assert "label (string) [required]" in spec
    assert "score (integer) [required]" in spec


def test_render_compact_schema_spec_guards_self_referential_schema() -> None:
    schema = {
        "type": "object",
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "children": {"type": "array", "items": {"$ref": "#/$defs/Node"}},
                },
                "required": ["label"],
            }
        },
        "properties": {"root": {"$ref": "#/$defs/Node"}},
        "required": ["root"],
    }
    spec = render_compact_schema_spec(schema)
    # Recursion guard renders the recursive ref once, not infinitely.
    assert spec.count("children (array of object)") == 1
    assert len(spec) <= COMPACT_SCHEMA_SPEC_MAX_CHARS


def test_render_compact_schema_spec_degraded_renders_nested_object_def_once() -> None:
    # Force degraded mode (full expansion over the cap) with an array element
    # (`Item`) whose field references a SECOND object def (`Detail`). The nested
    # def's fields must still appear in the prompt, exactly once.
    filler = {f"pad_field_{i:03d}": {"type": "string"} for i in range(300)}
    schema = {
        "type": "object",
        "$defs": {
            "Detail": {
                "type": "object",
                "properties": {
                    "detail_code": {"type": "string"},
                    "detail_weight": {"type": "integer"},
                },
                "required": ["detail_code"],
            },
            "Item": {
                "type": "object",
                "properties": {
                    **filler,
                    "detail": {"$ref": "#/$defs/Detail"},
                },
                "required": ["detail"],
            },
        },
        "properties": {"items": {"type": "array", "items": {"$ref": "#/$defs/Item"}}},
        "required": ["items"],
    }

    spec = render_compact_schema_spec(schema)

    # The full expansion exceeds the cap, so the renderer degraded.
    assert "items (array of object) -> Item [required]" in spec
    assert "\nItem:" in spec
    # Without the nested-def fix, `Detail`'s section and its fields would be
    # silently dropped (Item rendered with render_object_fields=False).
    assert "detail (object) -> Detail [required]" in spec
    assert spec.count("\nDetail:") == 1
    assert spec.count("detail_code (string)") == 1
    assert spec.count("detail_weight (integer)") == 1
