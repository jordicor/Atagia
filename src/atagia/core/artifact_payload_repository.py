"""SQLite persistence helpers for deduplicated artifact payload bytes."""

from __future__ import annotations

from typing import Any

from atagia.core.ids import generate_prefixed_id
from atagia.core.repositories import BaseRepository


class ArtifactPayloadRepository(BaseRepository):
    """Persistence operations for shared artifact payload blobs."""

    async def get_payload_blob(self, payload_blob_id: str, user_id: str) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM artifact_payload_blobs
            WHERE id = ?
              AND user_id = ?
            """,
            (payload_blob_id, user_id),
        )

    async def get_payload_for_artifact(self, artifact_id: str, user_id: str) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT
                apb.id AS payload_blob_id,
                a.id AS artifact_id,
                apb.user_id,
                apb.storage_kind,
                apb.blob_bytes,
                apb.storage_key,
                apb.external_uri,
                CASE
                    WHEN apb.storage_kind = 'external_ref' THEN apb.external_uri
                    ELSE apb.storage_key
                END AS storage_uri,
                apb.byte_size,
                apb.content_sha256 AS sha256,
                apb.content_sha256,
                apb.identity_kind,
                apb.status,
                apb.created_at,
                apb.updated_at
            FROM artifacts AS a
            JOIN artifact_payload_blobs AS apb ON apb.id = a.payload_blob_id
            WHERE a.id = ?
              AND a.user_id = ?
              AND apb.user_id = a.user_id
              AND apb.status = 'ready'
            """,
            (artifact_id, user_id),
        )

    async def find_active_content_payload(
        self,
        *,
        user_id: str,
        storage_kind: str,
        content_sha256: str,
        byte_size: int,
    ) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM artifact_payload_blobs
            WHERE user_id = ?
              AND storage_kind = ?
              AND identity_kind = 'content_sha256'
              AND content_sha256 = ?
              AND byte_size = ?
              AND status IN ('pending', 'ready', 'gc_pending')
            ORDER BY created_at ASC, id ASC
            LIMIT 1
            """,
            (user_id, storage_kind, content_sha256, byte_size),
        )

    async def find_active_external_payload(
        self,
        *,
        user_id: str,
        external_uri: str,
    ) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM artifact_payload_blobs
            WHERE user_id = ?
              AND identity_kind = 'external_uri'
              AND external_uri = ?
              AND status IN ('pending', 'ready', 'gc_pending')
            ORDER BY created_at ASC, id ASC
            LIMIT 1
            """,
            (user_id, external_uri),
        )

    async def create_payload_blob(
        self,
        *,
        payload_blob_id: str | None = None,
        user_id: str,
        storage_kind: str,
        identity_kind: str,
        content_sha256: str | None,
        byte_size: int,
        blob_bytes: bytes | None,
        storage_key: str | None,
        external_uri: str | None,
        status: str = "ready",
        commit: bool = True,
    ) -> dict[str, Any]:
        resolved_id = payload_blob_id or generate_prefixed_id("apb")
        timestamp = self._timestamp()
        await self._connection.execute(
            """
            INSERT INTO artifact_payload_blobs(
                id,
                user_id,
                storage_kind,
                identity_kind,
                content_sha256,
                byte_size,
                blob_bytes,
                storage_key,
                external_uri,
                status,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                resolved_id,
                user_id,
                storage_kind,
                identity_kind,
                content_sha256,
                int(byte_size),
                blob_bytes,
                storage_key,
                external_uri,
                status,
                timestamp,
                timestamp,
            ),
        )
        if commit:
            await self._connection.commit()
        created = await self.get_payload_blob(resolved_id, user_id)
        if created is None:
            raise RuntimeError(f"Failed to create artifact payload blob {resolved_id}")
        return created

    async def payload_has_live_artifacts(self, payload_blob_id: str, user_id: str) -> bool:
        row = await self._fetch_one(
            """
            SELECT 1 AS found
            FROM artifacts
            WHERE user_id = ?
              AND payload_blob_id = ?
              AND status NOT IN ('deleted', 'purged')
            LIMIT 1
            """,
            (user_id, payload_blob_id),
        )
        return row is not None
