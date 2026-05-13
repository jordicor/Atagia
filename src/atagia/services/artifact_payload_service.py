"""Deduplicated artifact payload persistence."""

from __future__ import annotations

import hashlib
import sqlite3
from typing import Any

import aiosqlite

from atagia.core.artifact_payload_repository import ArtifactPayloadRepository
from atagia.core.clock import Clock
from atagia.services.artifact_blob_store import ArtifactBlobStore


class ArtifactPayloadService:
    """Persist raw artifact payloads as user-scoped shared blobs."""

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

    async def get_or_create_payload_blob(
        self,
        blob: dict[str, Any] | None,
        *,
        user_id: str,
    ) -> dict[str, Any] | None:
        if blob is None:
            return None
        storage_kind = str(blob["storage_kind"])
        if storage_kind == "external_ref":
            return await self._get_or_create_external_payload(blob, user_id=user_id)
        return await self._get_or_create_content_payload(blob, user_id=user_id)

    async def _get_or_create_content_payload(
        self,
        blob: dict[str, Any],
        *,
        user_id: str,
    ) -> dict[str, Any]:
        storage_kind = str(blob["storage_kind"])
        blob_bytes = blob.get("blob_bytes")
        content_sha256 = self._content_sha256(blob)
        byte_size = self._byte_size(blob, blob_bytes=blob_bytes)
        storage_key = blob.get("storage_uri")
        payload_bytes = bytes(blob_bytes) if blob_bytes is not None else None
        if blob_bytes is not None:
            self._validate_content_identity(
                payload_bytes,
                expected_sha256=content_sha256,
                expected_byte_size=byte_size,
            )
        repository = ArtifactPayloadRepository(self._connection, self._clock)
        existing = await repository.find_active_content_payload(
            user_id=user_id,
            storage_kind=storage_kind,
            content_sha256=content_sha256,
            byte_size=byte_size,
        )
        if existing is not None:
            if (
                storage_kind == "local_file"
                and payload_bytes is not None
                and existing.get("storage_kind") == "local_file"
            ):
                if self._blob_store is None:
                    raise ValueError("Local artifact blob storage is not configured")
                stored = self._blob_store.store_bytes(user_id=user_id, content_bytes=payload_bytes)
                if stored.storage_uri != existing.get("storage_key"):
                    raise ValueError("Artifact payload storage key mismatch")
            return existing

        if storage_kind == "sqlite_blob":
            if payload_bytes is None:
                raise ValueError("SQLite artifact payload is missing bytes")
            return await self._create_with_content_race_retry(
                repository,
                user_id=user_id,
                storage_kind="sqlite_blob",
                content_sha256=content_sha256,
                byte_size=byte_size,
                blob_bytes=payload_bytes,
                storage_key=None,
            )
        if storage_kind != "local_file":
            raise ValueError(f"Unsupported artifact payload storage kind: {storage_kind}")
        if payload_bytes is not None:
            if self._blob_store is None:
                raise ValueError("Local artifact blob storage is not configured")
            stored = self._blob_store.store_bytes(user_id=user_id, content_bytes=payload_bytes)
            storage_key = stored.storage_uri
            byte_size = stored.byte_size
            content_sha256 = stored.sha256
        if not storage_key:
            raise ValueError("Local artifact payload is missing a storage key")
        return await self._create_with_content_race_retry(
            repository,
            user_id=user_id,
            storage_kind="local_file",
            content_sha256=content_sha256,
            byte_size=byte_size,
            blob_bytes=None,
            storage_key=str(storage_key),
        )

    async def _get_or_create_external_payload(
        self,
        blob: dict[str, Any],
        *,
        user_id: str,
    ) -> dict[str, Any]:
        external_uri = blob.get("storage_uri")
        if not external_uri:
            raise ValueError("External artifact payload is missing a URI")
        external_uri = str(external_uri)
        content_sha256 = str(blob.get("sha256") or hashlib.sha256(external_uri.encode("utf-8")).hexdigest())
        repository = ArtifactPayloadRepository(self._connection, self._clock)
        existing = await repository.find_active_external_payload(user_id=user_id, external_uri=external_uri)
        if existing is not None:
            return existing
        try:
            return await repository.create_payload_blob(
                user_id=user_id,
                storage_kind="external_ref",
                identity_kind="external_uri",
                content_sha256=content_sha256,
                byte_size=int(blob.get("byte_size") or 0),
                blob_bytes=None,
                storage_key=None,
                external_uri=external_uri,
                commit=False,
            )
        except sqlite3.IntegrityError:
            existing = await repository.find_active_external_payload(user_id=user_id, external_uri=external_uri)
            if existing is None:
                raise
            return existing

    async def _create_with_content_race_retry(
        self,
        repository: ArtifactPayloadRepository,
        *,
        user_id: str,
        storage_kind: str,
        content_sha256: str,
        byte_size: int,
        blob_bytes: bytes | None,
        storage_key: str | None,
    ) -> dict[str, Any]:
        try:
            return await repository.create_payload_blob(
                user_id=user_id,
                storage_kind=storage_kind,
                identity_kind="content_sha256",
                content_sha256=content_sha256,
                byte_size=byte_size,
                blob_bytes=blob_bytes,
                storage_key=storage_key,
                external_uri=None,
                commit=False,
            )
        except sqlite3.IntegrityError:
            existing = await repository.find_active_content_payload(
                user_id=user_id,
                storage_kind=storage_kind,
                content_sha256=content_sha256,
                byte_size=byte_size,
            )
            if existing is None:
                raise
            return existing

    @staticmethod
    def _content_sha256(blob: dict[str, Any]) -> str:
        explicit = blob.get("sha256")
        if explicit:
            return str(explicit)
        blob_bytes = blob.get("blob_bytes")
        if blob_bytes is None:
            raise ValueError("Content artifact payload is missing a sha256")
        return hashlib.sha256(bytes(blob_bytes)).hexdigest()

    @staticmethod
    def _byte_size(blob: dict[str, Any], *, blob_bytes: Any) -> int:
        explicit = blob.get("byte_size")
        if explicit is not None:
            return int(explicit)
        if blob_bytes is not None:
            return len(bytes(blob_bytes))
        return 0

    @staticmethod
    def _validate_content_identity(
        content_bytes: bytes,
        *,
        expected_sha256: str,
        expected_byte_size: int,
    ) -> None:
        if len(content_bytes) != expected_byte_size:
            raise ValueError("Artifact payload byte size mismatch")
        actual_sha256 = hashlib.sha256(content_bytes).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError("Artifact payload hash mismatch")
