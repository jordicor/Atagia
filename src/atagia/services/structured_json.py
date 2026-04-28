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

