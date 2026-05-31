"""Text-free validation artifacts for initial-context package rollout."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from atagia.services.chat_support import estimate_tokens

PROMPT_DIFF_ARTIFACT_SCHEMA_VERSION = 1
PREPARED_CONTEXT_MARKER = "[Prepared Initial Context]"

_SAFE_DIAGNOSTIC_KEYS = {
    "enabled",
    "rendered",
    "read_ms",
    "budget_tokens",
    "tokens_estimate",
    "selected_profile_items",
    "dropped_profile_items",
    "selected_curated_items",
    "dropped_curated_items",
    "deduped_source_refs",
    "dropped_sections",
    "known_empty",
    "overflow_dropped",
    "disabled_reason",
    "error",
}

_SAFE_PACKAGE_KEYS = {
    "package_kind",
    "status",
    "expected_key_hash",
    "package_id",
    "package_key_hash",
    "package_version",
    "updated_at",
    "fallback_reason",
}

_SAFE_PACKAGE_KINDS = {"baseline", "conversation", "none"}
_SAFE_PACKAGE_STATUSES = {
    "hit",
    "miss",
    "stale",
    "deleted",
    "unavailable",
    "signature_mismatch",
    "version_mismatch",
    "coordinate_incomplete",
    "failed",
}


def build_initial_context_package_prompt_diff_artifact(
    *,
    label: str,
    prompt_without_package: str,
    prompt_with_package: str,
    diagnostics_without_package: Mapping[str, Any] | None = None,
    diagnostics_with_package: Mapping[str, Any] | None = None,
    llm_calls_without_package: int | None = None,
    llm_calls_with_package: int | None = None,
    sql_statements_without_package: int | None = None,
    sql_statements_with_package: int | None = None,
    context_assembly_ms_without_package: float | None = None,
    context_assembly_ms_with_package: float | None = None,
) -> dict[str, Any]:
    """Build a private-rollout prompt diff without storing raw prompt text."""

    without_summary = _prompt_summary(prompt_without_package)
    with_summary = _prompt_summary(prompt_with_package)
    artifact: dict[str, Any] = {
        "schema_version": PROMPT_DIFF_ARTIFACT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label": _text_fingerprint(label),
        "text_free": True,
        "prompt": {
            "without_package": without_summary,
            "with_package": with_summary,
            "delta": {
                "chars": with_summary["chars"] - without_summary["chars"],
                "tokens_estimate": (
                    with_summary["tokens_estimate"]
                    - without_summary["tokens_estimate"]
                ),
                "prepared_context_marker_added": (
                    not without_summary["has_prepared_initial_context"]
                    and with_summary["has_prepared_initial_context"]
                ),
            },
        },
        "initial_context_package": {
            "without_package": sanitize_initial_context_package_diagnostics(
                diagnostics_without_package or {}
            ),
            "with_package": sanitize_initial_context_package_diagnostics(
                diagnostics_with_package or {}
            ),
        },
        "request_path": _request_path_summary(
            llm_calls_without_package=llm_calls_without_package,
            llm_calls_with_package=llm_calls_with_package,
            sql_statements_without_package=sql_statements_without_package,
            sql_statements_with_package=sql_statements_with_package,
            context_assembly_ms_without_package=(
                context_assembly_ms_without_package
            ),
            context_assembly_ms_with_package=context_assembly_ms_with_package,
        ),
    }
    artifact["rollout_checks"] = _rollout_checks(artifact)
    return artifact


def sanitize_initial_context_package_diagnostics(
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    """Return diagnostics safe for tracked logs and private prompt-diff summaries."""

    sanitized: dict[str, Any] = {}
    for key in _SAFE_DIAGNOSTIC_KEYS:
        if key not in diagnostics:
            continue
        sanitized[key] = _safe_top_level_diagnostic_value(key, diagnostics[key])
    packages = diagnostics.get("packages")
    if isinstance(packages, list):
        sanitized["packages"] = [
            _safe_package_diagnostics(package)
            for package in packages
            if isinstance(package, Mapping)
        ]
    else:
        sanitized["packages"] = []
    return sanitized


def write_initial_context_package_prompt_diff_artifact(
    artifact: Mapping[str, Any],
    output_dir: Path | str,
    *,
    filename: str | None = None,
) -> Path:
    """Write a text-free prompt-diff artifact to an ignored local output path."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    resolved_filename = filename or _artifact_filename(artifact.get("label"))
    artifact_path = output_path / resolved_filename
    artifact_path.write_text(
        json.dumps(dict(artifact), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return artifact_path


def _prompt_summary(prompt: str) -> dict[str, Any]:
    return {
        "sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "chars": len(prompt),
        "tokens_estimate": estimate_tokens(prompt),
        "has_prepared_initial_context": PREPARED_CONTEXT_MARKER in prompt,
    }


def _request_path_summary(
    *,
    llm_calls_without_package: int | None,
    llm_calls_with_package: int | None,
    sql_statements_without_package: int | None,
    sql_statements_with_package: int | None,
    context_assembly_ms_without_package: float | None,
    context_assembly_ms_with_package: float | None,
) -> dict[str, Any]:
    return {
        "llm_calls_without_package": llm_calls_without_package,
        "llm_calls_with_package": llm_calls_with_package,
        "llm_call_delta": _delta(llm_calls_without_package, llm_calls_with_package),
        "sql_statements_without_package": sql_statements_without_package,
        "sql_statements_with_package": sql_statements_with_package,
        "sql_statement_delta": _delta(
            sql_statements_without_package,
            sql_statements_with_package,
        ),
        "context_assembly_ms_without_package": context_assembly_ms_without_package,
        "context_assembly_ms_with_package": context_assembly_ms_with_package,
        "context_assembly_ms_delta": _delta(
            context_assembly_ms_without_package,
            context_assembly_ms_with_package,
        ),
    }


def _rollout_checks(artifact: Mapping[str, Any]) -> dict[str, Any]:
    request_path = artifact.get("request_path")
    if not isinstance(request_path, Mapping):
        request_path = {}
    package = artifact.get("initial_context_package")
    if not isinstance(package, Mapping):
        package = {}
    with_package = package.get("with_package")
    if not isinstance(with_package, Mapping):
        with_package = {}
    return {
        "package_read_added_llm_call": (
            request_path.get("llm_call_delta") is not None
            and request_path.get("llm_call_delta") > 0
        ),
        "package_rendered": bool(with_package.get("rendered")),
        "package_overflow_dropped": bool(with_package.get("overflow_dropped")),
        "package_statuses": [
            package_diag.get("status")
            for package_diag in with_package.get("packages", [])
            if isinstance(package_diag, Mapping)
        ],
    }


def _safe_top_level_diagnostic_value(key: str, value: Any) -> Any:
    if key in {
        "enabled",
        "rendered",
        "read_ms",
        "budget_tokens",
        "tokens_estimate",
        "selected_profile_items",
        "dropped_profile_items",
        "selected_curated_items",
        "dropped_curated_items",
        "deduped_source_refs",
        "overflow_dropped",
    }:
        return _safe_number_or_bool(value)
    if key in {"disabled_reason", "error"}:
        return _safe_string_fingerprint(value)
    if key == "dropped_sections":
        return [_safe_string_fingerprint(item) for item in value] if isinstance(value, list) else []
    if key == "known_empty" and isinstance(value, Mapping):
        return [
            {
                "key": _safe_string_fingerprint(raw_key),
                "value": bool(raw_value),
            }
            for raw_key, raw_value in value.items()
        ]
    return _safe_diagnostic_value(value)


def _safe_package_diagnostics(package: Mapping[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key in _SAFE_PACKAGE_KEYS:
        if key not in package:
            continue
        value = package[key]
        if key == "package_kind":
            sanitized[key] = _safe_enum(value, _SAFE_PACKAGE_KINDS)
        elif key == "status":
            sanitized[key] = _safe_enum(value, _SAFE_PACKAGE_STATUSES)
        elif key == "package_version":
            sanitized[key] = value if isinstance(value, int) else None
        elif key == "updated_at":
            sanitized[key] = _safe_string_fingerprint(value)
        elif key == "fallback_reason":
            sanitized[key] = _safe_string_fingerprint(value)
        else:
            sanitized[key] = _safe_string_fingerprint(value)
    return sanitized


def _safe_number_or_bool(value: Any) -> int | float | bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    return None


def _safe_enum(value: Any, allowed: set[str]) -> str | dict[str, Any] | None:
    if not isinstance(value, str):
        return None
    return value if value in allowed else _text_fingerprint(value)


def _safe_string_fingerprint(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return _text_fingerprint(str(value))


def _safe_diagnostic_value(value: Any) -> Any:
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, str):
        return _text_fingerprint(value)
    if isinstance(value, list):
        return [_safe_diagnostic_value(item) for item in value]
    if isinstance(value, Mapping):
        return {
            _text_fingerprint(str(key))["sha256"]: _safe_diagnostic_value(item)
            for key, item in value.items()
            if isinstance(key, (str, int, float, bool))
        }
    return _text_fingerprint(str(value))


def _text_fingerprint(text: str) -> dict[str, Any]:
    return {
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "chars": len(text),
    }


def _delta(before: int | float | None, after: int | float | None) -> int | float | None:
    if before is None or after is None:
        return None
    return after - before


def _artifact_filename(label: Any) -> str:
    label_hash = ""
    if isinstance(label, Mapping) and isinstance(label.get("sha256"), str):
        label_hash = label["sha256"][:12]
    safe_label = f"run-{label_hash}" if label_hash else "run"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{safe_label}-{timestamp}.json"
