"""Local, opt-in capture of normal Atagia operations and provider attempts."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
import hashlib
import logging
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable, Iterator
from uuid import uuid4

from atagia.diagnostics.contract import SCHEMA_VERSION, canonical_json_bytes, sha256_hex

logger = logging.getLogger(__name__)
_operation: ContextVar[tuple["DiagnosticRecorder", str, str] | None] = ContextVar("atagia_diagnostic_operation", default=None)
_attempt: ContextVar[tuple["DiagnosticRecorder", str] | None] = ContextVar("atagia_diagnostic_attempt", default=None)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _code_revision() -> str | None:
    try:
        repository = Path(__file__).resolve().parents[3]
        if not (repository / ".git").exists():
            return None
        result = subprocess.run(["git", "-c", f"safe.directory={repository.as_posix()}", "rev-parse", "HEAD"], cwd=repository, capture_output=True, text=True, timeout=2, check=True)
        revision = result.stdout.strip()
        return revision if len(revision) == 40 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _source_fingerprint() -> str | None:
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    try:
        for path in sorted(package.rglob("*.py")):
            digest.update(path.relative_to(package).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()
    except OSError:
        return None


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump(mode="json"))
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported diagnostic value: {type(value).__name__}")


class DiagnosticRecorder:
    """One writer per private session; failures latch and never alter provider results."""

    def __init__(self, directory: str | Path, *, max_blob_bytes: int = 1_048_576, max_session_bytes: int = 104_857_600) -> None:
        if max_blob_bytes <= 0 or max_session_bytes <= 0:
            raise ValueError("Diagnostic capture limits must be positive")
        parent = Path(directory).expanduser().resolve()
        if not parent.is_dir():
            raise ValueError(f"Diagnostic capture directory does not exist: {parent}")
        if shutil.disk_usage(parent).free < max_blob_bytes:
            raise ValueError("Diagnostic capture directory has insufficient free space")
        self.root = parent / f"capture-{uuid4().hex}"
        self.root.mkdir(mode=0o700)
        (self.root / "blobs").mkdir(mode=0o700)
        self.capture_id = self.root.name
        self.max_blob_bytes = max_blob_bytes
        self.max_session_bytes = max_session_bytes
        self.event_count = 0
        self.bytes_written = 0
        self.failed = False
        self.closed = False
        self._events_sha256 = hashlib.sha256()
        (self.root / "events.jsonl").touch()
        self._manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "capture_id": self.capture_id,
            "created_at": _utc_now(),
            "finished_at": None,
            "status": "incomplete",
            "event_count": 0,
            "events_sha256": None,
            "limits": {"max_blob_bytes": max_blob_bytes, "max_session_bytes": max_session_bytes},
            "failure_reason": None,
            "code_revision": _code_revision(),
            "source_fingerprint_sha256": _source_fingerprint(),
        }
        self._write_manifest()

    def _write_manifest(self) -> None:
        temporary = self.root / "manifest.tmp"
        temporary.write_bytes(canonical_json_bytes(self._manifest))
        temporary.replace(self.root / "manifest.json")

    def _fail(self, exc: BaseException) -> None:
        self.failed = True
        self._manifest["status"] = "failed"
        self._manifest["failure_reason"] = f"{type(exc).__name__}: {exc}"
        logger.error("Diagnostic capture failed at %s: %s", self.root, self._manifest["failure_reason"])
        try:
            self._write_manifest()
        except OSError:
            logger.exception("Diagnostic capture failure manifest could not be written at %s", self.root)

    def blob(self, value: bytes | str | Any) -> dict[str, Any] | None:
        if self.failed or self.closed:
            return None
        try:
            data = value if isinstance(value, bytes) else value.encode("utf-8") if isinstance(value, str) else canonical_json_bytes(_json_safe(value))
            if len(data) > self.max_blob_bytes:
                raise ValueError("blob exceeds max_blob_bytes; capture is not reproducible")
            digest = sha256_hex(data)
            target = self.root / "blobs" / digest
            if not target.exists():
                if self.bytes_written + len(data) > self.max_session_bytes:
                    raise ValueError("capture exceeds max_session_bytes; capture is not reproducible")
                target.write_bytes(data)
                self.bytes_written += len(data)
            return {"sha256": digest, "size_bytes": len(data), "encoding": "utf-8"}
        except Exception as exc:
            self._fail(exc if isinstance(exc, (OSError, TypeError, ValueError)) else ValueError(f"serialization failure: {type(exc).__name__}"))
            return None

    def event(self, kind: str, *, operation_id: str, trace_id: str, status: str, data: dict[str, Any], attempt_id: str | None = None, phase: str | None = None, parent_operation_id: str | None = None, purpose: str | None = None, component: str | None = None, card: str | None = None, user_id: str | None = None, turn_id: str | None = None, job_id: str | None = None) -> None:
        if self.failed or self.closed:
            return
        try:
            record: dict[str, Any] = {"seq": self.event_count + 1, "timestamp": _utc_now(), "kind": kind, "trace_id": trace_id, "operation_id": operation_id, "parent_operation_id": parent_operation_id, "attempt_id": attempt_id, "purpose": purpose, "component": component, "card": card, "user_id": user_id, "turn_id": turn_id, "job_id": job_id, "status": status, "data": _json_safe(data)}
            if phase is not None:
                record["phase"] = phase
            encoded = canonical_json_bytes(record)
            if self.bytes_written + len(encoded) > self.max_session_bytes:
                raise ValueError("capture exceeds max_session_bytes; capture is not reproducible")
            with (self.root / "events.jsonl").open("ab") as stream:
                stream.write(encoded)
            self._events_sha256.update(encoded)
            self.bytes_written += len(encoded)
            self.event_count += 1
        except Exception as exc:
            self._fail(exc if isinstance(exc, (OSError, TypeError, ValueError)) else ValueError(f"serialization failure: {type(exc).__name__}"))

    @contextmanager
    def operation(self, purpose: str, *, component: str | None = None, card: str | None = None, user_id: str | None = None, turn_id: str | None = None, job_id: str | None = None, input_data: dict[str, Any] | None = None) -> Iterator[str]:
        parent = _operation.get()
        if parent is not None and parent[0] is not self:
            parent = None
        operation_id = uuid4().hex
        trace_id = parent[1] if parent else uuid4().hex
        self.event("operation_start", trace_id=trace_id, operation_id=operation_id, parent_operation_id=parent[2] if parent else None, purpose=purpose, component=component, card=card, user_id=user_id, turn_id=turn_id, job_id=job_id, status="started", data=input_data or {})
        token = _operation.set((self, trace_id, operation_id))
        try:
            yield operation_id
        except BaseException as exc:
            self.event("operation_end", trace_id=trace_id, operation_id=operation_id, parent_operation_id=parent[2] if parent else None, purpose=purpose, component=component, card=card, user_id=user_id, turn_id=turn_id, job_id=job_id, status="cancelled" if not isinstance(exc, Exception) else "failure", data={"error_type": type(exc).__name__})
            raise
        else:
            self.event("operation_end", trace_id=trace_id, operation_id=operation_id, parent_operation_id=parent[2] if parent else None, purpose=purpose, component=component, card=card, user_id=user_id, turn_id=turn_id, job_id=job_id, status="success", data={})
        finally:
            _operation.reset(token)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.failed:
            return
        try:
            self._manifest.update({"finished_at": _utc_now(), "status": "complete", "event_count": self.event_count, "events_sha256": self._events_sha256.hexdigest()})
            self._write_manifest()
        except OSError as exc:
            self._fail(exc)

    def no_call(self, purpose: str, *, component: str, user_id: str | None = None, data: dict[str, Any] | None = None) -> None:
        """Record a deterministic/cache step without inventing model usage."""
        parent = current_operation(self)
        self.event("no_call", trace_id=parent[0] if parent else uuid4().hex, operation_id=uuid4().hex, parent_operation_id=parent[1] if parent else None, purpose=purpose, component=component, user_id=user_id, status="success", data=data or {})


def current_operation(recorder: DiagnosticRecorder | None = None) -> tuple[str, str] | None:
    active = _operation.get()
    if active is None or recorder is not None and active[0] is not recorder:
        return None
    return active[1], active[2]


@contextmanager
def bind_attempt(recorder: DiagnosticRecorder, attempt_id: str) -> Iterator[None]:
    token = _attempt.set((recorder, attempt_id))
    try:
        yield
    finally:
        _attempt.reset(token)


def capture_sent_payload(payload: Any) -> None:
    active = _attempt.get()
    if active is not None:
        recorder, attempt_id = active
        if isinstance(payload, dict):
            forbidden = {"headers", "extra_headers", "authorization", "api_key", "credentials"}
            if forbidden.intersection(payload):
                recorder._fail(ValueError("provider payload contains a credential container"))
                return
        ref = recorder.blob(payload)
        operation = current_operation(recorder)
        if ref is not None and operation is not None:
            recorder.event("provider_payload", trace_id=operation[0], operation_id=operation[1], attempt_id=attempt_id, status="sent", data={"sent_payload": ref})


def capture_raw_response(response: Any) -> None:
    active = _attempt.get()
    if active is not None:
        recorder, attempt_id = active
        ref = recorder.blob(response)
        operation = current_operation(recorder)
        if ref is not None and operation is not None:
            recorder.event("provider_raw", trace_id=operation[0], operation_id=operation[1], attempt_id=attempt_id, status="received", data={"raw_response": ref})


def capture_chat_operation(function: Callable[..., Any]) -> Callable[..., Any]:
    """Correlate a full chat turn, including retrieval, answer and effects."""
    @wraps(function)
    async def wrapped(service: Any, *args: Any, **kwargs: Any) -> Any:
        recorder = getattr(service.runtime.llm_client, "_diagnostic_recorder", None)
        if recorder is None:
            return await function(service, *args, **kwargs)
        user_id = args[0] if len(args) > 0 else kwargs["user_id"]
        conversation_id = args[1] if len(args) > 1 else kwargs["conversation_id"]
        message_text = args[2] if len(args) > 2 else kwargs["message_text"]
        with recorder.operation("chat_turn", component="chat", user_id=user_id, input_data={"conversation_id": conversation_id, "message": recorder.blob(message_text)}):
            return await function(service, *args, **kwargs)
    return wrapped


def capture_evidence_operation(function: Callable[..., Any]) -> Callable[..., Any]:
    """Correlate independent evidence decisions with source effects."""
    @wraps(function)
    async def wrapped(llm_client: Any, *args: Any, **kwargs: Any) -> Any:
        recorder = getattr(llm_client, "_diagnostic_recorder", None)
        if recorder is None:
            return await function(llm_client, *args, **kwargs)
        context = kwargs["context"]
        with recorder.operation("memory_extraction_evidence", component="extractor", card="evidence", user_id=context.user_id, input_data={"source": recorder.blob(kwargs["message_text"])}):
            return await function(llm_client, *args, **kwargs)
    return wrapped
