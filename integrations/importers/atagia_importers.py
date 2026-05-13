"""Offline importers for copyable Atagia platform integration bundles."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


@dataclass(frozen=True, slots=True)
class ImportMessage:
    """Normalized text message ready for Atagia ingestion."""

    role: str
    text: str
    source_seq: int
    message_id: str
    occurred_at: str | None = None


@dataclass(slots=True)
class ImportSummary:
    """Import result for CLI/tests."""

    source: str
    imported: int = 0
    skipped: int = 0
    errors: list[str] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "imported": self.imported,
            "skipped": self.skipped,
            "errors": list(self.errors or []),
        }


class AtagiaImportClient:
    """Small stdlib HTTP client for Atagia sidecar message backfill."""

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
        role: str,
        text: str,
        platform_id: str,
        message_id: str,
        source_seq: int,
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
            f"{self.base_url}/v1/conversations/{quote(conversation_id, safe='')}/messages",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-Atagia-User-Id": user_id,
                "X-Atagia-Conversation-Id": conversation_id,
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
    platform_id: str = "sillytavern",
    mode: str = "companion",
    character_id: str | None = None,
    user_persona_id: str | None = None,
    memory_privacy_mode: str = "balanced",
) -> ImportSummary:
    """Import a SillyTavern `.jsonl` chat export."""
    records = _read_jsonl_records(source)
    messages = []
    for index, record in enumerate(records, start=1):
        role = _role_from_record(record)
        text = _text_from_record(record)
        if role not in {"user", "assistant"} or not text:
            continue
        messages.append(
            _message(
                source_name="sillytavern",
                conversation_id=conversation_id,
                role=role,
                text=text,
                index=index,
                occurred_at=_optional_text(record.get("send_date") or record.get("timestamp")),
            )
        )
    return _ingest_messages(
        "sillytavern_jsonl",
        messages,
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
    platform_id: str = "sillytavern",
    mode: str = "companion",
    character_id: str | None = None,
    user_persona_id: str | None = None,
    memory_privacy_mode: str = "balanced",
) -> ImportSummary:
    """Import plain text lorebook entries as admin-reviewed backfill."""
    text = _read_text(source)
    entries = [entry.strip() for entry in text.split("\n\n") if entry.strip()]
    messages = [
        _message(
            source_name="sillytavern_lorebook",
            conversation_id=conversation_id,
            role="user",
            text=entry,
            index=index,
        )
        for index, entry in enumerate(entries, start=1)
    ]
    return _ingest_messages(
        "sillytavern_lorebook",
        messages,
        client=client,
        user_id=user_id,
        conversation_id=conversation_id,
        platform_id=platform_id,
        mode=mode,
        character_id=character_id,
        user_persona_id=user_persona_id,
        memory_privacy_mode=memory_privacy_mode,
    )


def import_openclaw_session(
    source: str | Path | dict[str, Any],
    *,
    client: Any,
    user_id: str,
    conversation_id: str,
    platform_id: str = "openclaw",
    mode: str = "general_qa",
    character_id: str | None = None,
    user_persona_id: str | None = None,
    memory_privacy_mode: str = "balanced",
) -> ImportSummary:
    """Import an OpenClaw transcript shaped as messages/transcript/sessionFile."""
    payload = _read_json_object(source)
    records = _extract_records(payload, "messages", "transcript", "sessionFile.messages")
    messages = _records_to_messages("openclaw", conversation_id, records)
    return _ingest_messages(
        "openclaw_session",
        messages,
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
    platform_id: str = "hermes",
    mode: str = "general_qa",
    character_id: str | None = None,
    user_persona_id: str | None = None,
    memory_privacy_mode: str = "balanced",
) -> ImportSummary:
    """Import Hermes session/memory exports when shaped as text records."""
    payload = _read_json_object(source)
    records = _extract_records(payload, "messages", "transcript", "memories")
    messages = _records_to_messages("hermes", conversation_id, records)
    return _ingest_messages(
        "hermes_export",
        messages,
        client=client,
        user_id=user_id,
        conversation_id=conversation_id,
        platform_id=platform_id,
        mode=mode,
        character_id=character_id,
        user_persona_id=user_persona_id,
        memory_privacy_mode=memory_privacy_mode,
    )


def _ingest_messages(
    source: str,
    messages: Iterable[ImportMessage],
    *,
    client: Any,
    user_id: str,
    conversation_id: str,
    platform_id: str,
    mode: str | None,
    character_id: str | None,
    user_persona_id: str | None,
    memory_privacy_mode: str,
) -> ImportSummary:
    summary = ImportSummary(source=source, imported=0, skipped=0, errors=[])
    for message in messages:
        if not message.text:
            summary.skipped += 1
            continue
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
            summary.errors.append(str(exc))
    return summary


def _read_jsonl_records(source: str | Path) -> list[dict[str, Any]]:
    records = []
    for line in _read_text(source).splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if isinstance(record, dict):
            records.append(record)
    return records


def _read_json_object(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(source, dict):
        return source
    loaded = json.loads(_read_text(source))
    return loaded if isinstance(loaded, dict) else {"messages": loaded}


def _read_text(source: str | Path) -> str:
    path = Path(source)
    try:
        exists = path.exists()
    except OSError:
        exists = False
    if exists:
        return path.read_text(encoding="utf-8")
    return str(source)


def _extract_records(payload: dict[str, Any], *paths: str) -> list[dict[str, Any]]:
    for path in paths:
        value: Any = payload
        for part in path.split("."):
            if isinstance(value, dict):
                value = value.get(part)
            else:
                value = None
                break
        if isinstance(value, list):
            return [record for record in value if isinstance(record, dict)]
    return []


def _records_to_messages(
    source_name: str,
    conversation_id: str,
    records: list[dict[str, Any]],
) -> list[ImportMessage]:
    messages = []
    for index, record in enumerate(records, start=1):
        role = _role_from_record(record)
        text = _text_from_record(record)
        if role not in {"user", "assistant"} or not text:
            continue
        messages.append(
            _message(
                source_name=source_name,
                conversation_id=conversation_id,
                role=role,
                text=text,
                index=index,
                occurred_at=_optional_text(
                    record.get("occurred_at")
                    or record.get("created_at")
                    or record.get("timestamp")
                    or record.get("send_date")
                ),
            )
        )
    return messages


def _message(
    *,
    source_name: str,
    conversation_id: str,
    role: str,
    text: str,
    index: int,
    occurred_at: str | None = None,
) -> ImportMessage:
    normalized_text = text.strip()
    source_seq = _source_seq(index, role, normalized_text)
    message_id = f"{source_name}:{conversation_id}:{role}:{index}:{_short_hash(normalized_text)}"
    return ImportMessage(
        role=role,
        text=normalized_text,
        source_seq=source_seq,
        message_id=message_id,
        occurred_at=occurred_at,
    )


def _role_from_record(record: dict[str, Any]) -> str:
    role = str(record.get("role") or "").strip().lower()
    if role in {"user", "assistant"}:
        return role
    if record.get("is_user") is True:
        return "user"
    if record.get("is_user") is False:
        return "assistant"
    author = str(record.get("author") or record.get("name") or "").lower()
    if author in {"user", "human"}:
        return "user"
    return "assistant"


def _text_from_record(record: dict[str, Any]) -> str:
    for key in ("mes", "content", "text", "message"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _source_seq(index: int, role: str, text: str) -> int:
    role_offset = 50000 if role == "assistant" else 1
    content_offset = int(_short_hash(text)[:8], 16) % 49999
    return (index * 100000) + role_offset + content_offset


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _optional_text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
