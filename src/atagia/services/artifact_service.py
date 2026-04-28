"""Attachment planning and persistence for first-class artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import base64
import binascii
import hashlib
from typing import Any

import aiosqlite

from atagia.core.artifact_repository import ArtifactRepository
from atagia.core.clock import Clock
from atagia.core.ids import generate_prefixed_id
from atagia.memory.context_composer import ContextComposer
from atagia.models.schemas_api import AttachmentInput
from atagia.services.artifact_blob_store import ArtifactBlobStore

_ATTACHMENT_PROMPT_HEADER = "[Attachments omitted]"
_ATTACHMENT_PLACEHOLDER_HEADER = "[Artifact omitted]"
_MAX_ATTACHMENT_PROMPT_PREVIEW_CHARS = 220
_MAX_ATTACHMENT_CHUNK_TOKENS = 280
_MAX_ATTACHMENT_CHUNK_CHARS = 1800


@dataclass(frozen=True, slots=True)
class PreparedArtifact:
    """Fully normalized artifact data ready for persistence."""

    artifact: dict[str, Any]
    blob: dict[str, Any] | None
    chunks: list[dict[str, Any]]
    link: dict[str, Any]
    prompt_placeholder: str


@dataclass(frozen=True, slots=True)
class AttachmentBundle:
    """Normalized attachment payload for one chat turn."""

    prompt_text: str
    context_placeholder: str | None
    attachments: list[dict[str, Any]]
    artifacts: list[PreparedArtifact]


@dataclass(frozen=True, slots=True)
class ArtifactPayload:
    """User-scoped artifact serving payload with conservative raw access gates."""

    artifact: dict[str, Any]
    storage_kind: str | None
    content_bytes: bytes | None
    storage_uri: str | None
    byte_size: int
    sha256: str | None
    raw_available: bool
    raw_returned: bool
    raw_block_reason: str | None


class ArtifactService:
    """Plan, store, and surface artifact-backed attachments."""

    def __init__(
        self,
        connection: aiosqlite.Connection,
        clock: Clock,
        *,
        blob_store: ArtifactBlobStore | None = None,
    ) -> None:
        self._connection = connection
        self._clock = clock
        self._blob_store = blob_store

    def prepare_attachments(
        self,
        *,
        message_text: str,
        attachments: list[AttachmentInput | dict[str, Any]] | None,
        user_id: str,
        conversation: dict[str, Any],
    ) -> AttachmentBundle:
        normalized = [
            item if isinstance(item, AttachmentInput) else AttachmentInput.model_validate(item)
            for item in (attachments or [])
        ]
        if not normalized:
            return AttachmentBundle(
                prompt_text=message_text,
                context_placeholder=None,
                attachments=[],
                artifacts=[],
            )

        prepared_artifacts = [
            self._prepare_artifact(
                attachment=attachment,
                user_id=user_id,
                conversation=conversation,
                ordinal=ordinal,
            )
            for ordinal, attachment in enumerate(normalized)
        ]
        placeholder_lines = [
            prepared.prompt_placeholder for prepared in prepared_artifacts
        ]
        prompt_block = "\n".join([_ATTACHMENT_PROMPT_HEADER, *placeholder_lines])
        prompt_text = message_text.strip()
        if prompt_text:
            prompt_text = f"{prompt_text}\n\n{prompt_block}"
        else:
            prompt_text = prompt_block
        return AttachmentBundle(
            prompt_text=prompt_text,
            context_placeholder="\n".join([_ATTACHMENT_PLACEHOLDER_HEADER, *placeholder_lines]),
            attachments=[self._attachment_metadata(prepared) for prepared in prepared_artifacts],
            artifacts=prepared_artifacts,
        )

    async def persist_prepared_attachments(
        self,
        *,
        bundle: AttachmentBundle,
        message_id: str,
        commit: bool = True,
    ) -> list[dict[str, Any]]:
        repository = ArtifactRepository(self._connection, self._clock)
        created: list[dict[str, Any]] = []
        for prepared in bundle.artifacts:
            artifact = prepared.artifact
            blob = self._blob_for_persistence(prepared.blob, user_id=str(artifact["user_id"]))
            created_artifact = await repository.create_artifact(
                artifact_id=str(artifact["id"]),
                user_id=str(artifact["user_id"]),
                workspace_id=artifact.get("workspace_id"),
                conversation_id=artifact.get("conversation_id"),
                message_id=message_id,
                artifact_type=str(artifact["artifact_type"]),
                source_kind=str(artifact["source_kind"]),
                source_ref=artifact.get("source_ref"),
                mime_type=artifact.get("mime_type"),
                filename=artifact.get("filename"),
                title=artifact.get("title"),
                content_hash=artifact.get("content_hash"),
                size_bytes=artifact.get("size_bytes"),
                page_count=artifact.get("page_count"),
                status=str(artifact.get("status", "ready")),
                privacy_level=int(artifact.get("privacy_level", 0)),
                preserve_verbatim=bool(artifact.get("preserve_verbatim", False)),
                skip_raw_by_default=bool(artifact.get("skip_raw_by_default", True)),
                requires_explicit_request=bool(artifact.get("requires_explicit_request", True)),
                metadata_json=dict(artifact.get("metadata_json") or {}),
                summary_text=artifact.get("summary_text"),
                index_text=artifact.get("index_text"),
                storage_kind=blob.get("storage_kind") if blob else None,
                blob_bytes=blob.get("blob_bytes") if blob else None,
                storage_uri=blob.get("storage_uri") if blob else None,
                blob_byte_size=blob.get("byte_size") if blob else None,
                blob_sha256=blob.get("sha256") if blob else None,
                commit=False,
            )
            artifact_id = str(created_artifact["id"])
            if prepared.chunks:
                chunk_rows = [
                    {
                        "id": chunk.get("id") or generate_prefixed_id("arc"),
                        "artifact_id": artifact_id,
                        "user_id": str(artifact["user_id"]),
                        "chunk_index": int(chunk["chunk_index"]),
                        "source_start_offset": chunk.get("source_start_offset"),
                        "source_end_offset": chunk.get("source_end_offset"),
                        "text": str(chunk["text"]),
                        "token_count": int(chunk["token_count"]),
                        "kind": str(chunk["kind"]),
                    }
                    for chunk in prepared.chunks
                ]
                await repository.create_artifact_chunks_bulk(chunk_rows, commit=False)
            await repository.create_artifact_link(
                user_id=str(artifact["user_id"]),
                message_id=message_id,
                artifact_id=artifact_id,
                relation_kind=str(prepared.link.get("relation_kind", "attachment")),
                ordinal=int(prepared.link.get("ordinal", 0)),
                link_id=str(prepared.link["id"]),
                commit=False,
            )
            created.append(created_artifact)
        if commit:
            await self._connection.commit()
        return created

    async def fetch_artifact_payload(
        self,
        *,
        user_id: str,
        artifact_id: str,
        include_raw: bool = False,
    ) -> ArtifactPayload | None:
        repository = ArtifactRepository(self._connection, self._clock)
        artifact = await repository.get_artifact(artifact_id, user_id)
        if artifact is None or artifact.get("status") in {"deleted", "purged"}:
            return None
        blob = await repository.get_artifact_blob(artifact_id, user_id)
        storage_kind = str(blob["storage_kind"]) if blob is not None else None
        raw_available = blob is not None and (
            blob.get("blob_bytes") is not None or blob.get("storage_uri") is not None
        )
        raw_block_reason = self._raw_block_reason(artifact, include_raw, raw_available)
        content_bytes = None
        storage_uri = None
        if raw_available and raw_block_reason is None:
            if storage_kind == "local_file":
                if self._blob_store is None:
                    raw_block_reason = "local_file_store_unavailable"
                else:
                    content_bytes = self._blob_store.read_bytes(str(blob["storage_uri"]))
            else:
                content_bytes = blob.get("blob_bytes")
                storage_uri = blob.get("storage_uri")
            if content_bytes is not None:
                self._validate_blob_hash(content_bytes, str(blob["sha256"]))
        return ArtifactPayload(
            artifact=artifact,
            storage_kind=storage_kind,
            content_bytes=content_bytes,
            storage_uri=storage_uri,
            byte_size=int(blob["byte_size"]) if blob is not None else 0,
            sha256=str(blob["sha256"]) if blob is not None else None,
            raw_available=raw_available,
            raw_returned=raw_available and raw_block_reason is None,
            raw_block_reason=raw_block_reason,
        )

    def _prepare_artifact(
        self,
        *,
        attachment: AttachmentInput,
        user_id: str,
        conversation: dict[str, Any],
        ordinal: int,
    ) -> PreparedArtifact:
        artifact_id = generate_prefixed_id("art")
        decoded_bytes = self._decode_base64_payload(attachment.content_base64)
        text_payload = self._extract_text_payload(attachment, decoded_bytes)
        storage_kind, blob_bytes, storage_uri, byte_size, content_hash = self._storage_payload(
            attachment,
            decoded_bytes=decoded_bytes,
            text_payload=text_payload,
        )
        summary_text = self._build_summary_text(attachment, artifact_id, text_payload, byte_size)
        index_text = self._build_index_text(attachment, summary_text, text_payload)
        chunks = self._build_chunks(
            artifact_id=artifact_id,
            user_id=user_id,
            attachment=attachment,
            text_payload=text_payload,
            summary_text=summary_text,
        )
        prompt_placeholder = self._build_prompt_placeholder(
            attachment=attachment,
            artifact_id=artifact_id,
            text_payload=text_payload,
            byte_size=byte_size,
            summary_text=summary_text,
            ordinal=ordinal,
        )
        metadata = dict(attachment.metadata)
        metadata.setdefault("relevance_state", "active_work_material")
        metadata.setdefault("relevance_source", "attachment_ingest")
        return PreparedArtifact(
            artifact={
                "id": artifact_id,
                "user_id": user_id,
                "workspace_id": conversation.get("workspace_id"),
                "conversation_id": conversation.get("id"),
                "message_id": None,
                "artifact_type": attachment.kind,
                "source_kind": self._source_kind(attachment),
                "source_ref": self._source_ref(attachment),
                "mime_type": attachment.mime_type,
                "filename": attachment.filename,
                "title": attachment.title,
                "content_hash": content_hash,
                "size_bytes": byte_size,
                "page_count": attachment.page_count,
                "status": "ready",
                "privacy_level": attachment.privacy_level,
                "preserve_verbatim": attachment.preserve_verbatim,
                "skip_raw_by_default": attachment.skip_raw_by_default,
                "requires_explicit_request": attachment.requires_explicit_request,
                "metadata_json": metadata,
                "summary_text": summary_text,
                "index_text": index_text,
            },
            blob={
                "storage_kind": storage_kind,
                "blob_bytes": blob_bytes,
                "storage_uri": storage_uri,
                "byte_size": byte_size,
                "sha256": content_hash,
            }
            if storage_kind is not None
            else None,
            chunks=chunks,
            link={
                "id": generate_prefixed_id("arl"),
                "user_id": user_id,
                "message_id": "",
                "artifact_id": artifact_id,
                "relation_kind": "attachment",
                "ordinal": ordinal,
            },
            prompt_placeholder=prompt_placeholder,
        )

    @staticmethod
    def _raw_block_reason(
        artifact: dict[str, Any],
        include_raw: bool,
        raw_available: bool,
    ) -> str | None:
        if not raw_available:
            return None
        if include_raw:
            return None
        if bool(artifact.get("requires_explicit_request")):
            return "explicit_request_required"
        if bool(artifact.get("skip_raw_by_default")):
            return "skip_raw_by_default"
        return "raw_not_requested"

    def _blob_for_persistence(
        self,
        blob: dict[str, Any] | None,
        *,
        user_id: str,
    ) -> dict[str, Any] | None:
        if blob is None:
            return None
        if blob.get("storage_kind") != "local_file" or blob.get("blob_bytes") is None:
            return blob
        if self._blob_store is None:
            raise ValueError("Local artifact blob storage is not configured")
        stored = self._blob_store.store_bytes(user_id=user_id, content_bytes=bytes(blob["blob_bytes"]))
        return {
            **blob,
            "storage_kind": stored.storage_kind,
            "blob_bytes": stored.blob_bytes,
            "storage_uri": stored.storage_uri,
            "byte_size": stored.byte_size,
            "sha256": stored.sha256,
        }

    @staticmethod
    def _validate_blob_hash(content_bytes: bytes, expected_sha256: str) -> None:
        actual_sha256 = hashlib.sha256(content_bytes).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError("Artifact blob hash mismatch")

    @staticmethod
    def _source_kind(attachment: AttachmentInput) -> str:
        if attachment.url is not None:
            return "url"
        if attachment.content_base64 is not None:
            return "base64"
        if attachment.content_text is not None:
            return "pasted_text"
        if attachment.source_ref is not None:
            return "external_ref"
        return "host_embedded"

    @staticmethod
    def _source_ref(attachment: AttachmentInput) -> str | None:
        return attachment.source_ref or attachment.url or attachment.filename or attachment.title

    @staticmethod
    def _decode_base64_payload(content_base64: str | None) -> bytes | None:
        if content_base64 is None:
            return None
        normalized = "".join(content_base64.split())
        if not normalized:
            return None
        try:
            return base64.b64decode(normalized, validate=True)
        except (ValueError, binascii.Error):
            raise ValueError("Invalid base64 attachment payload") from None

    def _extract_text_payload(
        self,
        attachment: AttachmentInput,
        decoded_bytes: bytes | None,
    ) -> str | None:
        if attachment.content_text is not None:
            return self._normalize_text(attachment.content_text)
        if decoded_bytes is None:
            return None
        if attachment.mime_type and attachment.mime_type.startswith("text/"):
            return self._try_decode_utf8(decoded_bytes)
        if attachment.kind in {"pasted_text", "file", "base64"}:
            decoded_text = self._try_decode_utf8(decoded_bytes)
            if decoded_text is not None:
                return decoded_text
        return None

    @staticmethod
    def _try_decode_utf8(raw_bytes: bytes) -> str | None:
        try:
            return ArtifactService._normalize_text(raw_bytes.decode("utf-8"))
        except UnicodeDecodeError:
            return None

    def _storage_payload(
        self,
        attachment: AttachmentInput,
        *,
        decoded_bytes: bytes | None,
        text_payload: str | None,
    ) -> tuple[str | None, bytes | None, str | None, int, str]:
        if attachment.url is not None and decoded_bytes is None and text_payload is None:
            storage_uri = attachment.url
            sha256 = hashlib.sha256(storage_uri.encode("utf-8")).hexdigest()
            return "external_ref", None, storage_uri, 0, sha256
        if decoded_bytes is not None:
            sha256 = hashlib.sha256(decoded_bytes).hexdigest()
            if self._blob_store is not None:
                return "local_file", decoded_bytes, None, len(decoded_bytes), sha256
            return "sqlite_blob", decoded_bytes, None, len(decoded_bytes), sha256
        if text_payload is not None:
            encoded = text_payload.encode("utf-8")
            sha256 = hashlib.sha256(encoded).hexdigest()
            if self._blob_store is not None:
                return "local_file", encoded, None, len(encoded), sha256
            return "sqlite_blob", encoded, None, len(encoded), sha256
        reference = self._source_ref(attachment)
        if reference is not None:
            sha256 = hashlib.sha256(reference.encode("utf-8")).hexdigest()
            return "external_ref", None, reference, 0, sha256
        sha256 = hashlib.sha256(generate_prefixed_id("art").encode("utf-8")).hexdigest()
        return None, None, None, 0, sha256

    def _build_summary_text(
        self,
        attachment: AttachmentInput,
        artifact_id: str,
        text_payload: str | None,
        byte_size: int,
    ) -> str:
        parts = [
            f"{attachment.kind} attachment",
            f"artifact_id={artifact_id}",
        ]
        if attachment.title:
            parts.append(f"title={self._truncate_field(attachment.title, 80)}")
        if attachment.filename:
            parts.append(f"filename={self._truncate_field(attachment.filename, 80)}")
        if attachment.mime_type:
            parts.append(f"mime={self._truncate_field(attachment.mime_type, 80)}")
        if attachment.page_count is not None:
            parts.append(f"pages={attachment.page_count}")
        if byte_size:
            parts.append(f"size={byte_size} bytes")
        if attachment.preserve_verbatim:
            parts.append("verbatim")
        if attachment.privacy_level:
            parts.append(f"privacy={attachment.privacy_level}")
        if text_payload:
            parts.append(f"preview={self._truncate_field(text_payload, _MAX_ATTACHMENT_PROMPT_PREVIEW_CHARS)}")
        elif attachment.source_ref:
            parts.append(f"ref={self._truncate_field(attachment.source_ref, 120)}")
        return "; ".join(parts)

    def _build_index_text(
        self,
        attachment: AttachmentInput,
        summary_text: str,
        text_payload: str | None,
    ) -> str:
        parts = [summary_text]
        for value in (attachment.title, attachment.filename, attachment.mime_type, attachment.source_ref):
            if value:
                parts.append(self._truncate_field(value, 120))
        if text_payload:
            parts.append(self._truncate_field(text_payload, _MAX_ATTACHMENT_PROMPT_PREVIEW_CHARS))
        return " ".join(part for part in parts if part).strip()

    def _build_prompt_placeholder(
        self,
        *,
        attachment: AttachmentInput,
        artifact_id: str,
        text_payload: str | None,
        byte_size: int,
        summary_text: str,
        ordinal: int,
    ) -> str:
        parts = [
            f"- artifact_id={artifact_id}",
            f"type={attachment.kind}",
            f"source={self._source_kind(attachment)}",
            f"ordinal={ordinal + 1}",
        ]
        if attachment.title:
            parts.append(f"title={self._truncate_field(attachment.title, 80)}")
        if attachment.filename:
            parts.append(f"filename={self._truncate_field(attachment.filename, 80)}")
        if attachment.mime_type:
            parts.append(f"mime={self._truncate_field(attachment.mime_type, 80)}")
        if attachment.page_count is not None:
            parts.append(f"pages={attachment.page_count}")
        if byte_size:
            parts.append(f"size={byte_size} bytes")
        if attachment.preserve_verbatim:
            parts.append("verbatim")
        if text_payload:
            parts.append("summary available")
        elif attachment.url:
            parts.append("reference only")
        return " ".join([self._truncate_field(summary_text, 240), *parts]).strip()

    def _build_chunks(
        self,
        *,
        artifact_id: str,
        user_id: str,
        attachment: AttachmentInput,
        text_payload: str | None,
        summary_text: str,
    ) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = [
            {
                "id": generate_prefixed_id("arc"),
                "artifact_id": artifact_id,
                "user_id": user_id,
                "chunk_index": 0,
                "source_start_offset": None,
                "source_end_offset": None,
                "text": summary_text,
                "token_count": self._estimate_tokens(summary_text),
                "kind": "summary",
            }
        ]
        if text_payload is None:
            return chunks
        for chunk_index, chunk_text in enumerate(self._chunk_text(text_payload), start=1):
            chunks.append(
                {
                    "id": generate_prefixed_id("arc"),
                    "artifact_id": artifact_id,
                    "user_id": user_id,
                    "chunk_index": chunk_index,
                    "source_start_offset": None,
                    "source_end_offset": None,
                    "text": chunk_text,
                    "token_count": self._estimate_tokens(chunk_text),
                    "kind": "parsed" if attachment.content_text is not None else "extracted",
                }
            )
        return chunks

    @staticmethod
    def _chunk_text(text: str) -> list[str]:
        normalized = ArtifactService._normalize_text(text)
        if not normalized:
            return []

        lines = normalized.split("\n")
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        def flush_current() -> None:
            nonlocal current, current_tokens
            if current:
                chunk_text = "\n".join(current).strip()
                if chunk_text:
                    chunks.append(chunk_text)
            current = []
            current_tokens = 0

        for line in lines:
            piece = line.strip()
            if not piece:
                flush_current()
                continue
            piece_tokens = ArtifactService._estimate_tokens(piece)
            if piece_tokens > _MAX_ATTACHMENT_CHUNK_TOKENS or len(piece) > _MAX_ATTACHMENT_CHUNK_CHARS:
                flush_current()
                chunks.extend(
                    ArtifactService._split_long_piece(piece)
                )
                continue
            if current and current_tokens + piece_tokens > _MAX_ATTACHMENT_CHUNK_TOKENS:
                flush_current()
            current.append(piece)
            current_tokens += piece_tokens

        flush_current()
        return chunks or [normalized]

    @staticmethod
    def _split_long_piece(piece: str) -> list[str]:
        chunks: list[str] = []
        start = 0
        while start < len(piece):
            end = min(len(piece), start + _MAX_ATTACHMENT_CHUNK_CHARS)
            chunks.append(piece[start:end].strip())
            start = end
        return [chunk for chunk in chunks if chunk]

    @staticmethod
    def _truncate_field(value: str, limit: int) -> str:
        normalized = " ".join(value.split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: max(0, limit - 3)].rstrip() + "..."

    @staticmethod
    def _normalize_text(value: str) -> str:
        return value.replace("\r\n", "\n").replace("\r", "\n").strip()

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return ContextComposer.estimate_tokens(text)

    @staticmethod
    def _attachment_metadata(prepared: PreparedArtifact) -> dict[str, Any]:
        artifact = prepared.artifact
        artifact_metadata = artifact.get("metadata_json")
        if not isinstance(artifact_metadata, dict):
            artifact_metadata = {}
        return {
            "artifact_id": artifact["id"],
            "artifact_type": artifact["artifact_type"],
            "source_kind": artifact["source_kind"],
            "source_ref": artifact.get("source_ref"),
            "mime_type": artifact.get("mime_type"),
            "filename": artifact.get("filename"),
            "title": artifact.get("title"),
            "content_hash": artifact.get("content_hash"),
            "size_bytes": artifact.get("size_bytes"),
            "page_count": artifact.get("page_count"),
            "privacy_level": artifact.get("privacy_level", 0),
            "preserve_verbatim": artifact.get("preserve_verbatim", False),
            "skip_raw_by_default": artifact.get("skip_raw_by_default", True),
            "requires_explicit_request": artifact.get("requires_explicit_request", True),
            "relevance_state": artifact_metadata.get("relevance_state"),
            "relevance_source": artifact_metadata.get("relevance_source"),
            "status": artifact.get("status", "ready"),
            "summary_text": artifact.get("summary_text"),
            "index_text": artifact.get("index_text"),
        }
