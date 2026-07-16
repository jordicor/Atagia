"""Offline transcript importers for copyable Atagia integration bundles.

Only host transcript rows with an explicit user/assistant role are eligible for
message ingestion.  Curated memory exports are intentionally reported and
skipped: importing those safely requires a separate, typed memory importer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from atagia.transport_ids import encode_path_id


_IDENTITY_SCHEMA = "atagia.external-message.v1"
_SUPPORTED_ROLES = frozenset({"user", "assistant"})


@dataclass(frozen=True, slots=True)
class ImportMessage:
    """Normalized transcript message ready for Atagia ingestion."""

    role: Literal["user", "assistant"]
    text: str
    source_seq: int | None
    message_id: str
    occurred_at: str | None = None


@dataclass(slots=True)
class ImportSummary:
    """Complete validation and ingestion report for one import batch."""

    source: str
    total_records: int = 0
    imported: int = 0
    failed: int = 0
    skipped_by_reason: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def skipped(self) -> int:
        return sum(self.skipped_by_reason.values())

    def skip(self, reason: str, detail: str, *, error: bool = False) -> None:
        self.skipped_by_reason[reason] = self.skipped_by_reason.get(reason, 0) + 1
        (self.errors if error else self.warnings).append(detail)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "total_records": self.total_records,
            "imported": self.imported,
            "failed": self.failed,
            "skipped": self.skipped,
            "skipped_by_reason": dict(self.skipped_by_reason),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


class ImportBatchValidationError(ValueError):
    """Raised before ingestion when strict batch validation finds bad rows."""

    def __init__(self, summary: ImportSummary) -> None:
        self.summary = summary
        super().__init__(
            f"{summary.source} validation failed with {len(summary.errors)} error(s)"
        )


@dataclass(frozen=True, slots=True)
class _InvalidRecord:
    reason: str


class AtagiaImportClient:
    """Small stdlib HTTP client for Atagia sidecar transcript backfill."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8100",
        api_key: str = "",
        timeout_seconds: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def ingest_message(
        self,
        *,
        user_id: str,
        conversation_id: str,
        role: Literal["user", "assistant"],
        text: str,
        platform_id: str,
        message_id: str,
        source_seq: int | None,
        mode: str | None = None,
        character_id: str | None = None,
        user_persona_id: str | None = None,
        occurred_at: str | None = None,
        memory_privacy_mode: str = "balanced",
    ) -> dict[str, Any]:
        payload = {
            "user_id": user_id,
            "role": role,
            "text": text,
            "platform_id": platform_id,
            "mode": mode,
            "character_id": character_id,
            "user_persona_id": user_persona_id,
            "message_id": message_id,
            "source_seq": source_seq,
            "occurred_at": occurred_at,
            "ingest_origin": "backfill",
            "confirmation_strategy": "admin_review_only",
            "memory_privacy_mode": memory_privacy_mode,
        }
        data = json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}/v1/conversations/{encode_path_id(conversation_id)}/messages",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-Atagia-User-Id": user_id,
                "X-Atagia-Platform-Id": platform_id,
                "X-Atagia-Ingest-Origin": "backfill",
                "X-Atagia-Confirmation-Strategy": "admin_review_only",
                "X-Atagia-Memory-Privacy-Mode": memory_privacy_mode,
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(str(exc.reason)) from exc
        return json.loads(raw) if raw else {}


def import_sillytavern_jsonl(
    source: str | Path,
    *,
    client: Any,
    user_id: str,
    conversation_id: str,
    host_installation_id: str,
    host_account_id: str,
    host_conversation_id: str,
    platform_id: str = "sillytavern",
    mode: str = "companion",
    character_id: str | None = None,
    user_persona_id: str | None = None,
    memory_privacy_mode: str = "balanced",
    strict: bool = False,
) -> ImportSummary:
    """Import a SillyTavern chat export with explicit ``is_user`` roles."""
    records = _read_jsonl_records(source)
    summary = ImportSummary(source="sillytavern_jsonl", total_records=len(records))
    messages = _records_to_messages(
        integration_kind="sillytavern",
        records=records,
        summary=summary,
        strict=strict,
        role_format="sillytavern",
        user_id=user_id,
        host_installation_id=host_installation_id,
        host_account_id=host_account_id,
        host_conversation_id=host_conversation_id,
    )
    return _finish_import(
        summary,
        messages,
        strict=strict,
        client=client,
        user_id=user_id,
        conversation_id=conversation_id,
        platform_id=platform_id,
        mode=mode,
        character_id=character_id,
        user_persona_id=user_persona_id,
        memory_privacy_mode=memory_privacy_mode,
    )


def import_sillytavern_lorebook_text(
    source: str | Path,
    *,
    client: Any,
    user_id: str,
    conversation_id: str,
    host_installation_id: str,
    host_account_id: str,
    host_conversation_id: str,
    platform_id: str = "sillytavern",
    mode: str = "companion",
    character_id: str | None = None,
    user_persona_id: str | None = None,
    memory_privacy_mode: str = "balanced",
    strict: bool = False,
) -> ImportSummary:
    """Report unsupported lorebook entries without fabricating chat turns.

    The otherwise-unused parameters deliberately mirror transcript importers so
    callers can record one uniform report.  No network request is made.
    """
    del (
        client,
        user_id,
        conversation_id,
        host_installation_id,
        host_account_id,
        host_conversation_id,
        platform_id,
        mode,
        character_id,
        user_persona_id,
        memory_privacy_mode,
        strict,
    )
    entries = [entry.strip() for entry in _read_text(source).split("\n\n") if entry.strip()]
    summary = ImportSummary(source="sillytavern_lorebook", total_records=len(entries))
    for index in range(1, len(entries) + 1):
        summary.skip(
            "curated_memory_unsupported",
            f"record {index}: lorebook entry skipped; curated-memory import is unsupported",
        )
    return summary


def import_openclaw_session(
    source: str | Path | dict[str, Any],
    *,
    client: Any,
    user_id: str,
    conversation_id: str,
    host_installation_id: str,
    host_account_id: str,
    host_conversation_id: str,
    platform_id: str = "openclaw",
    mode: str = "general_qa",
    character_id: str | None = None,
    user_persona_id: str | None = None,
    memory_privacy_mode: str = "balanced",
    strict: bool = False,
) -> ImportSummary:
    """Import an OpenClaw transcript shaped as messages/transcript/sessionFile."""
    payload = _read_json_object(source)
    records = _extract_records(payload, "messages", "transcript", "sessionFile.messages")
    summary = ImportSummary(source="openclaw_session", total_records=len(records))
    messages = _records_to_messages(
        integration_kind="openclaw",
        records=records,
        summary=summary,
        strict=strict,
        role_format="role",
        user_id=user_id,
        host_installation_id=host_installation_id,
        host_account_id=host_account_id,
        host_conversation_id=host_conversation_id,
    )
    return _finish_import(
        summary,
        messages,
        strict=strict,
        client=client,
        user_id=user_id,
        conversation_id=conversation_id,
        platform_id=platform_id,
        mode=mode,
        character_id=character_id,
        user_persona_id=user_persona_id,
        memory_privacy_mode=memory_privacy_mode,
    )


def import_hermes_export(
    source: str | Path | dict[str, Any],
    *,
    client: Any,
    user_id: str,
    conversation_id: str,
    host_installation_id: str,
    host_account_id: str,
    host_conversation_id: str,
    platform_id: str = "hermes",
    mode: str = "general_qa",
    character_id: str | None = None,
    user_persona_id: str | None = None,
    memory_privacy_mode: str = "balanced",
    strict: bool = False,
) -> ImportSummary:
    """Import only Hermes transcript rows; curated memories are reported."""
    payload = _read_json_object(source)
    records = _extract_records(payload, "messages", "transcript")
    curated = payload.get("memories")
    curated_records = curated if isinstance(curated, list) else []
    summary = ImportSummary(
        source="hermes_export",
        total_records=len(records) + len(curated_records),
    )
    for index in range(1, len(curated_records) + 1):
        summary.skip(
            "curated_memory_unsupported",
            f"memory {index}: curated Hermes memory skipped; dedicated import is unsupported",
        )
    messages = _records_to_messages(
        integration_kind="hermes",
        records=records,
        summary=summary,
        strict=strict,
        role_format="role",
        user_id=user_id,
        host_installation_id=host_installation_id,
        host_account_id=host_account_id,
        host_conversation_id=host_conversation_id,
    )
    return _finish_import(
        summary,
        messages,
        strict=strict,
        client=client,
        user_id=user_id,
        conversation_id=conversation_id,
        platform_id=platform_id,
        mode=mode,
        character_id=character_id,
        user_persona_id=user_persona_id,
        memory_privacy_mode=memory_privacy_mode,
    )


def canonical_external_message_id(
    *,
    integration_kind: str,
    host_installation_id: str,
    host_account_id: str,
    user_id: str,
    host_conversation_id: str,
    source_namespace: str,
    host_message_id: str,
    role: Literal["user", "assistant"],
    generation_id: str,
) -> str:
    """Hash a canonical host identity tuple; message content is never an input."""
    fields = {
        "schema": _IDENTITY_SCHEMA,
        "integration_kind": _required_identity(integration_kind, "integration_kind"),
        "host_installation_id": _required_identity(
            host_installation_id, "host_installation_id"
        ),
        "host_account_id": _required_identity(host_account_id, "host_account_id"),
        "atagia_user_id": _required_identity(user_id, "user_id"),
        "host_conversation_id": _required_identity(
            host_conversation_id, "host_conversation_id"
        ),
        "source_namespace": _required_identity(source_namespace, "source_namespace"),
        "host_message_id": _required_identity(host_message_id, "host_message_id"),
        "role": role,
        "generation_id": _required_identity(generation_id, "generation_id"),
    }
    if role not in _SUPPORTED_ROLES:
        raise ValueError(f"unsupported role: {role!r}")
    canonical = json.dumps(
        fields,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"extmsg_{hashlib.sha256(canonical).hexdigest()}"


def _finish_import(
    summary: ImportSummary,
    messages: Iterable[ImportMessage],
    *,
    strict: bool,
    client: Any,
    user_id: str,
    conversation_id: str,
    platform_id: str,
    mode: str | None,
    character_id: str | None,
    user_persona_id: str | None,
    memory_privacy_mode: str,
) -> ImportSummary:
    if strict and summary.errors:
        raise ImportBatchValidationError(summary)
    for message in messages:
        try:
            client.ingest_message(
                user_id=user_id,
                conversation_id=conversation_id,
                role=message.role,
                text=message.text,
                platform_id=platform_id,
                message_id=message.message_id,
                source_seq=message.source_seq,
                mode=mode,
                character_id=character_id,
                user_persona_id=user_persona_id,
                occurred_at=message.occurred_at,
                memory_privacy_mode=memory_privacy_mode,
            )
            summary.imported += 1
        except Exception as exc:
            summary.failed += 1
            summary.errors.append(
                f"source_seq {message.source_seq}: ingestion failed: {exc}"
            )
    return summary


def _records_to_messages(
    *,
    integration_kind: str,
    records: list[Any],
    summary: ImportSummary,
    strict: bool,
    role_format: Literal["role", "sillytavern"],
    user_id: str,
    host_installation_id: str,
    host_account_id: str,
    host_conversation_id: str,
) -> list[ImportMessage]:
    # Validate batch identity even when the transcript is empty.
    for name, value in (
        ("host_installation_id", host_installation_id),
        ("host_account_id", host_account_id),
        ("host_conversation_id", host_conversation_id),
        ("user_id", user_id),
    ):
        _required_identity(value, name)

    messages: list[ImportMessage] = []
    for index, record in enumerate(records, start=1):
        if isinstance(record, _InvalidRecord):
            summary.skip(
                record.reason,
                f"record {index}: {record.reason.replace('_', ' ')}",
                error=strict,
            )
            continue
        if not isinstance(record, dict):
            summary.skip(
                "invalid_record",
                f"record {index}: expected a JSON object",
                error=strict,
            )
            continue
        role, role_reason = _explicit_role(record, role_format=role_format)
        if role is None:
            summary.skip(
                role_reason,
                f"record {index}: {role_reason.replace('_', ' ')}",
                error=strict,
            )
            continue
        text = _text_from_record(record)
        if not text:
            summary.skip(
                "empty_content",
                f"record {index}: empty content skipped",
                error=strict,
            )
            continue
        source_namespace, host_message_id, generation_id = _host_identity(
            integration_kind=integration_kind,
            record=record,
            index=index,
        )
        message_id = canonical_external_message_id(
            integration_kind=integration_kind,
            host_installation_id=host_installation_id,
            host_account_id=host_account_id,
            user_id=user_id,
            host_conversation_id=host_conversation_id,
            source_namespace=source_namespace,
            host_message_id=host_message_id,
            role=role,
            generation_id=generation_id,
        )
        stored_id = _stored_atagia_message_id(record)
        if stored_id is not None and stored_id != message_id:
            summary.skip(
                "source_mapping_mismatch",
                f"record {index}: stored live mapping does not match this import identity",
                error=strict,
            )
            continue
        source_mapping = _source_mapping(record)
        mapped_source_seq = source_mapping.get("source_seq")
        if "source_seq" in source_mapping:
            if mapped_source_seq is None:
                source_seq = None
            elif (
                isinstance(mapped_source_seq, int)
                and not isinstance(mapped_source_seq, bool)
                and mapped_source_seq >= 1
            ):
                source_seq = mapped_source_seq
            else:
                summary.skip(
                    "source_mapping_mismatch",
                    f"record {index}: stored live source_seq is invalid",
                    error=strict,
                )
                continue
        else:
            source_seq = index
        messages.append(
            ImportMessage(
                role=role,
                text=text,
                source_seq=source_seq,
                message_id=message_id,
                occurred_at=_optional_text(
                    record.get("occurred_at")
                    or record.get("created_at")
                    or record.get("timestamp")
                    or record.get("send_date")
                ),
            )
        )
    return messages


def _read_jsonl_records(source: str | Path) -> list[Any]:
    records: list[Any] = []
    for line in _read_text(source).splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            records.append(_InvalidRecord("invalid_json"))
            continue
        if not isinstance(record, dict):
            records.append(_InvalidRecord("invalid_record"))
            continue
        records.append(record)
    return records


def _read_json_object(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(source, dict):
        return source
    loaded = json.loads(_read_text(source))
    if isinstance(loaded, dict):
        return loaded
    if isinstance(loaded, list):
        return {"messages": loaded}
    raise ValueError("expected a JSON object or transcript array")


def _read_text(source: str | Path) -> str:
    path = Path(source)
    try:
        exists = path.exists()
    except OSError:
        exists = False
    if exists:
        return path.read_text(encoding="utf-8")
    return str(source)


def _extract_records(payload: dict[str, Any], *paths: str) -> list[Any]:
    for path in paths:
        value: Any = payload
        for part in path.split("."):
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(part)
        if isinstance(value, list):
            return list(value)
    return []


def _explicit_role(
    record: dict[str, Any],
    *,
    role_format: Literal["role", "sillytavern"],
) -> tuple[Literal["user", "assistant"] | None, str]:
    if role_format == "sillytavern" and isinstance(record.get("is_user"), bool):
        return ("user" if record["is_user"] else "assistant"), ""
    if "role" not in record or not isinstance(record.get("role"), str):
        return None, "missing_role"
    role = record["role"].strip().lower()
    if role == "user":
        return "user", ""
    if role == "assistant":
        return "assistant", ""
    return None, "unsupported_role"


def _host_identity(
    *,
    integration_kind: str,
    record: dict[str, Any],
    index: int,
) -> tuple[str, str, str]:
    source_mapping = _source_mapping(record)
    mapped_host_id = _optional_scalar(source_mapping.get("host_message_id"))
    host_message_id = mapped_host_id or _first_scalar(
        record,
        "message_id",
        "messageId",
        "event_id",
        "eventId",
        "id",
    )
    source_namespace = "host_message" if host_message_id is not None else "backfill_message"
    if host_message_id is None:
        host_message_id = f"ordinal:{index}"

    mapped_generation = _optional_scalar(
        source_mapping.get("host_generation_id")
        or source_mapping.get("generation_id")
    )
    generation_id = mapped_generation or _first_scalar(
        record,
        "generation_id",
        "generationId",
        "swipe_id",
        "swipeId",
    )
    if integration_kind == "sillytavern" and generation_id is None:
        swipe_info = record.get("swipe_info")
        swipe_id = record.get("swipe_id")
        if isinstance(swipe_info, list) and isinstance(swipe_id, int):
            if 0 <= swipe_id < len(swipe_info) and isinstance(swipe_info[swipe_id], dict):
                generation_id = _first_scalar(
                    swipe_info[swipe_id], "gen_id", "generation_id", "id"
                )
    return source_namespace, host_message_id, generation_id or "default"


def _source_mapping(record: dict[str, Any]) -> dict[str, Any]:
    extra = record.get("extra")
    if not isinstance(extra, dict):
        return {}
    mapping = extra.get("atagia_source_identity")
    return mapping if isinstance(mapping, dict) else {}


def _stored_atagia_message_id(record: dict[str, Any]) -> str | None:
    return _optional_scalar(_source_mapping(record).get("atagia_message_id"))


def _text_from_record(record: dict[str, Any]) -> str:
    for key in ("mes", "content", "text", "message"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _first_scalar(record: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = _optional_scalar(record.get(key))
        if value is not None:
            return value
    return None


def _optional_scalar(value: Any) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (str, int)):
        normalized = str(value).strip()
        return normalized or None
    return None


def _required_identity(value: Any, name: str) -> str:
    normalized = _optional_scalar(value)
    if normalized is None:
        raise ValueError(f"{name} must be a non-empty string or integer")
    return normalized


def _optional_text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
