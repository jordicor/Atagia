"""Structured JSON recovery for LLM outputs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from atagia._vendor.ai_json_cleanroom import (
    ValidateOptions,
    ValidationIssue,
    ValidationResult,
    validate_ai_json,
)

COMPACT_SCHEMA_SPEC_MAX_CHARS = 4096
_COMPACT_SCHEMA_SPEC_INDENT = "  "
_COMPACT_SCHEMA_SPEC_MAX_ENUM_VALUES = 24


@dataclass(frozen=True, slots=True)
class StructuredJSONPayload:
    """Decoded JSON payload plus parser diagnostics."""

    data: Any
    source: str | None
    warnings: tuple[str, ...] = ()


class StructuredJSONDecodeError(ValueError):
    """Raised when an LLM response cannot be recovered as JSON."""

    def __init__(self, message: str, *, details: tuple[str, ...]) -> None:
        super().__init__(message)
        self.details = details


def decode_structured_json_payload(output_text: str) -> StructuredJSONPayload:
    """Recover one JSON object or array from a raw LLM response.

    The cleanroom layer handles mechanical recovery only: markdown fences,
    surrounding prose, truncation detection, and conservative syntax repairs.
    Domain validation remains the caller's responsibility.
    """

    result = _validate(output_text, normalize_curly_quotes="never")
    if result.json_valid:
        return _payload_from_result(result)

    if not result.likely_truncated:
        curly_result = _validate(output_text, normalize_curly_quotes="auto")
        if curly_result.json_valid:
            warnings = (
                *_warning_details(curly_result),
                "$: JSON was recovered after curly quote normalization.",
            )
            return StructuredJSONPayload(
                data=curly_result.data,
                source=_source(curly_result),
                warnings=warnings,
            )

    raise StructuredJSONDecodeError(
        "Response was not valid JSON.",
        details=_error_details(result),
    )


def _validate(output_text: str, *, normalize_curly_quotes: str) -> ValidationResult:
    return validate_ai_json(
        output_text,
        options=ValidateOptions(
            strict=True,
            extract_json=True,
            tolerate_trailing_commas=True,
            allow_json_in_code_fences=True,
            allow_bare_top_level_scalars=False,
            enable_safe_repairs=True,
            allow_json5_like=True,
            normalize_curly_quotes=normalize_curly_quotes,
        ),
    )


def _payload_from_result(result: ValidationResult) -> StructuredJSONPayload:
    return StructuredJSONPayload(
        data=result.data,
        source=_source(result),
        warnings=_warning_details(result),
    )


def _source(result: ValidationResult) -> str | None:
    source = result.info.get("source") if isinstance(result.info, dict) else None
    return str(source) if source else None


def _warning_details(result: ValidationResult) -> tuple[str, ...]:
    return tuple(_issue_detail(issue) for issue in result.warnings)


def _error_details(result: ValidationResult) -> tuple[str, ...]:
    details = ["$: Response was not valid JSON."]
    if result.likely_truncated:
        reasons = _truncation_reasons(result)
        if reasons:
            details.append(f"$: JSON output appears truncated ({', '.join(reasons)}).")
        else:
            details.append("$: JSON output appears truncated.")
    source = _source(result)
    if source:
        details.append(f"$: JSON extraction source was {source}.")
    details.extend(_issue_detail(issue) for issue in result.errors)
    return tuple(dict.fromkeys(details))


def _issue_detail(issue: ValidationIssue) -> str:
    path = issue.path or "$"
    code = issue.code.value if isinstance(issue.code, Enum) else str(issue.code)
    return f"{path}: {code}: {issue.message}"


def _truncation_reasons(result: ValidationResult) -> tuple[str, ...]:
    reasons: list[str] = []
    for issue in result.errors:
        detail = issue.detail or {}
        raw_reasons = detail.get("truncation_reasons")
        if isinstance(raw_reasons, list):
            reasons.extend(str(reason) for reason in raw_reasons if reason)
    return tuple(dict.fromkeys(reasons))


def render_compact_schema_spec(schema: dict[str, Any]) -> str:
    """Render a JSON Schema as a compact, human-readable field specification.

    The output lists one line per field (name, type, inline enum values, and a
    ``required``/``optional`` marker), indents nested objects, resolves
    ``$defs``/``$ref`` inline by name, and renders the fields of an
    array-of-objects element exactly once. It emits ONLY schema-derived tokens
    (field names, types, enum values); descriptions and any request/user data
    are excluded, so the spec carries no prompt-injection surface.

    Output is deterministic: properties follow the schema's declared order. If
    the full render exceeds ``COMPACT_SCHEMA_SPEC_MAX_CHARS`` it degrades to the
    top-level structure with each repeated element's fields rendered once.
    """

    defs = schema.get("$defs")
    defs_map = defs if isinstance(defs, dict) else {}
    full_lines = _compact_schema_lines(schema, defs=defs_map, depth=0, render_object_fields=True)
    full = "\n".join(full_lines)
    if len(full) <= COMPACT_SCHEMA_SPEC_MAX_CHARS:
        return full

    return _render_degraded_schema_spec(schema, defs=defs_map)


def _render_degraded_schema_spec(
    schema: dict[str, Any],
    *,
    defs: dict[str, Any],
) -> str:
    """Render the top-level structure plus each referenced element's fields once.

    Used when the fully expanded spec exceeds the size cap. Object element types
    are referenced by name at the top level, then defined once in a flat section
    so the model still sees every field exactly once without recursive blow-up.
    Object-typed fields discovered while rendering an element def enqueue their
    own ``$def`` for a flat section too, so nested object fields are not silently
    dropped. The ``referenced``/``referenced_set`` dedup bounds this to one
    section per def, so each def still renders at most once.
    """
    top_lines: list[str] = []
    referenced: list[str] = []
    referenced_set: set[str] = set()

    def _enqueue(ref_name: str | None, target: dict[str, Any] | None) -> str:
        """Record an object def for a flat section; return its ``-> RefName`` suffix."""
        if ref_name is None or target is None:
            return ""
        if ref_name not in referenced_set:
            referenced_set.add(ref_name)
            referenced.append(ref_name)
        return f" -> {ref_name}"

    properties = schema.get("properties")
    if isinstance(properties, dict):
        required = schema.get("required")
        required_set = set(required) if isinstance(required, list) else set()
        for field_name, field_schema in properties.items():
            if not isinstance(field_schema, dict):
                continue
            type_label = _compact_schema_type_label(field_schema, defs=defs)
            enum_suffix = _compact_schema_enum_suffix(field_schema, defs=defs)
            marker = "required" if field_name in required_set else "optional"
            suffix = _enqueue(
                _compact_schema_ref_name(field_schema, defs=defs),
                _compact_schema_object_target(field_schema, defs=defs),
            )
            top_lines.append(
                f"{field_name} ({type_label}){enum_suffix}{suffix} [{marker}]"
            )

    sections: list[str] = ["\n".join(top_lines)]
    # ``referenced`` may grow while iterating (a def's object-typed field enqueues
    # another def); index iteration keeps newly discovered defs in scope. The
    # dedup set bounds the loop so each def renders exactly one section.
    index = 0
    while index < len(referenced):
        ref_name = referenced[index]
        index += 1
        target = defs.get(ref_name)
        if not isinstance(target, dict):
            continue
        field_lines = _degraded_element_def_lines(
            target,
            defs=defs,
            enqueue=_enqueue,
        )
        sections.append("\n".join([f"{ref_name}:", *field_lines]))
    return "\n\n".join(section for section in sections if section)


def _degraded_element_def_lines(
    target: dict[str, Any],
    *,
    defs: dict[str, Any],
    enqueue: Any,
) -> list[str]:
    """Render one element def's fields (one level deep) for the degraded spec.

    Like ``_compact_schema_lines(..., render_object_fields=False)`` but, for each
    object-typed field, it enqueues that field's ``$def`` (via ``enqueue``) so the
    nested object's fields get their own flat section instead of being dropped.
    """
    resolved = _resolve_schema_ref(target, defs=defs)
    properties = resolved.get("properties")
    if not isinstance(properties, dict):
        return []
    required = resolved.get("required")
    required_set = set(required) if isinstance(required, list) else set()
    indent = _COMPACT_SCHEMA_SPEC_INDENT
    lines: list[str] = []
    for field_name, field_schema in properties.items():
        if not isinstance(field_schema, dict):
            continue
        type_label = _compact_schema_type_label(field_schema, defs=defs)
        enum_suffix = _compact_schema_enum_suffix(field_schema, defs=defs)
        marker = "required" if field_name in required_set else "optional"
        suffix = enqueue(
            _compact_schema_ref_name(field_schema, defs=defs),
            _compact_schema_object_target(field_schema, defs=defs),
        )
        lines.append(f"{indent}{field_name} ({type_label}){enum_suffix}{suffix} [{marker}]")
    return lines


def _resolve_schema_ref(node: dict[str, Any], *, defs: dict[str, Any]) -> dict[str, Any]:
    """Resolve a local ``$ref`` against ``$defs`` by name; return the node otherwise."""
    ref = node.get("$ref")
    if not isinstance(ref, str):
        return node
    name = ref.rsplit("/", 1)[-1]
    target = defs.get(name)
    if isinstance(target, dict):
        return target
    return node


def _compact_schema_type_label(node: dict[str, Any], *, defs: dict[str, Any]) -> str:
    """Return a short type label for a (possibly ref'd) schema node."""
    resolved = _resolve_schema_ref(node, defs=defs)
    if "enum" in resolved:
        base = resolved.get("type")
        return str(base) if isinstance(base, str) else "string"
    node_type = resolved.get("type")
    if isinstance(node_type, str):
        if node_type == "array":
            element = resolved.get("items")
            if isinstance(element, dict):
                element_resolved = _resolve_schema_ref(element, defs=defs)
                if element_resolved.get("type") == "object" and isinstance(
                    element_resolved.get("properties"), dict
                ):
                    return "array of object"
                return f"array of {_compact_schema_type_label(element, defs=defs)}"
            return "array"
        return node_type
    if isinstance(node_type, list):
        non_null = [item for item in node_type if item != "null"]
        if non_null:
            return "|".join(str(item) for item in non_null)
    variants = resolved.get("anyOf") or resolved.get("oneOf")
    if isinstance(variants, list):
        labels = [
            _compact_schema_type_label(variant, defs=defs)
            for variant in variants
            if isinstance(variant, dict) and variant.get("type") != "null"
        ]
        deduped = list(dict.fromkeys(label for label in labels if label))
        if deduped:
            return "|".join(deduped)
    return "any"


def _compact_schema_enum_suffix(node: dict[str, Any], *, defs: dict[str, Any]) -> str:
    """Return an inline ``enum: a|b|c`` suffix, or an empty string."""
    resolved = _resolve_schema_ref(node, defs=defs)
    values = resolved.get("enum")
    if not isinstance(values, list) or not values:
        return ""
    rendered = [str(value) for value in values]
    if len(rendered) > _COMPACT_SCHEMA_SPEC_MAX_ENUM_VALUES:
        shown = rendered[:_COMPACT_SCHEMA_SPEC_MAX_ENUM_VALUES]
        return " enum: " + "|".join(shown) + f"|... (+{len(rendered) - len(shown)} more)"
    return " enum: " + "|".join(rendered)


def _compact_schema_object_target(
    node: dict[str, Any],
    *,
    defs: dict[str, Any],
) -> dict[str, Any] | None:
    """Return the object schema whose fields should be rendered, if any.

    Resolves refs, and for arrays returns the element schema when it is an
    object with properties (so array-of-objects renders the element's fields).
    """
    resolved = _resolve_schema_ref(node, defs=defs)
    if resolved.get("type") == "object" and isinstance(resolved.get("properties"), dict):
        return resolved
    if resolved.get("type") == "array":
        element = resolved.get("items")
        if isinstance(element, dict):
            element_resolved = _resolve_schema_ref(element, defs=defs)
            if element_resolved.get("type") == "object" and isinstance(
                element_resolved.get("properties"), dict
            ):
                return element_resolved
    return None


def _compact_schema_lines(
    node: dict[str, Any],
    *,
    defs: dict[str, Any],
    depth: int,
    render_object_fields: bool,
    seen_refs: tuple[str, ...] = (),
) -> list[str]:
    """Render one object node's properties into compact spec lines."""
    resolved = _resolve_schema_ref(node, defs=defs)
    properties = resolved.get("properties")
    if not isinstance(properties, dict):
        return []
    required = resolved.get("required")
    required_set = set(required) if isinstance(required, list) else set()
    indent = _COMPACT_SCHEMA_SPEC_INDENT * depth
    lines: list[str] = []
    for field_name, field_schema in properties.items():
        if not isinstance(field_schema, dict):
            continue
        type_label = _compact_schema_type_label(field_schema, defs=defs)
        enum_suffix = _compact_schema_enum_suffix(field_schema, defs=defs)
        marker = "required" if field_name in required_set else "optional"
        lines.append(f"{indent}{field_name} ({type_label}){enum_suffix} [{marker}]")

        if not render_object_fields:
            continue
        object_target = _compact_schema_object_target(field_schema, defs=defs)
        if object_target is None:
            continue
        ref_name = _compact_schema_ref_name(field_schema, defs=defs)
        if ref_name is not None and ref_name in seen_refs:
            continue
        next_seen = (*seen_refs, ref_name) if ref_name is not None else seen_refs
        lines.extend(
            _compact_schema_lines(
                object_target,
                defs=defs,
                depth=depth + 1,
                render_object_fields=True,
                seen_refs=next_seen,
            )
        )
    return lines


def _compact_schema_ref_name(node: dict[str, Any], *, defs: dict[str, Any]) -> str | None:
    """Return the ``$defs`` name a node (or its array element) refers to, if any."""
    ref = node.get("$ref")
    if isinstance(ref, str):
        return ref.rsplit("/", 1)[-1]
    if node.get("type") == "array":
        element = node.get("items")
        if isinstance(element, dict):
            element_ref = element.get("$ref")
            if isinstance(element_ref, str):
                return element_ref.rsplit("/", 1)[-1]
    return None

