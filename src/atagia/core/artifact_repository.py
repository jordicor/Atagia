"""SQLite persistence helpers for first-class artifacts and attachments."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

import aiosqlite

from atagia.core.clock import Clock
from atagia.core.ids import generate_prefixed_id
from atagia.core.repositories import BaseRepository, _decode_json_columns, _encode_json
from atagia.models.schemas_memory import MemoryScope


class ArtifactRepository(BaseRepository):
    """Persistence operations for artifacts, blobs, chunks, and links."""

    async def create_artifact(
        self,
        *,
        artifact_id: str | None = None,
        user_id: str,
        workspace_id: str | None,
        conversation_id: str | None,
        message_id: str | None,
        artifact_type: str,
        source_kind: str,
        source_ref: str | None = None,
        mime_type: str | None = None,
        filename: str | None = None,
        title: str | None = None,
        content_hash: str | None = None,
        size_bytes: int | None = None,
        page_count: int | None = None,
        status: str = "ready",
        privacy_level: int = 0,
        preserve_verbatim: bool = False,
        skip_raw_by_default: bool = True,
        requires_explicit_request: bool = True,
        metadata_json: dict[str, Any] | None = None,
        summary_text: str | None = None,
        index_text: str | None = None,
        storage_kind: str | None = None,
        blob_bytes: bytes | None = None,
        storage_uri: str | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        resolved_artifact_id = artifact_id or generate_prefixed_id("art")
        timestamp = self._timestamp()
        await self._connection.execute(
            """
            INSERT INTO artifacts(
                id,
                user_id,
                workspace_id,
                conversation_id,
                message_id,
                artifact_type,
                source_kind,
                source_ref,
                mime_type,
                filename,
                title,
                content_hash,
                size_bytes,
                page_count,
                status,
                privacy_level,
                preserve_verbatim,
                skip_raw_by_default,
                requires_explicit_request,
                metadata_json,
                summary_text,
                index_text,
                created_at,
                updated_at,
                deleted_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
            """,
            (
                resolved_artifact_id,
                user_id,
                workspace_id,
                conversation_id,
                message_id,
                artifact_type,
                source_kind,
                source_ref,
                mime_type,
                filename,
                title,
                content_hash,
                size_bytes,
                page_count,
                status,
                privacy_level,
                int(preserve_verbatim),
                int(skip_raw_by_default),
                int(requires_explicit_request),
                _encode_json(metadata_json),
                summary_text,
                index_text,
                timestamp,
                timestamp,
            ),
        )
        if blob_bytes is not None or storage_kind is not None or storage_uri is not None:
            resolved_storage_kind = storage_kind or ("external_ref" if storage_uri is not None else "sqlite_blob")
            await self._connection.execute(
                """
                INSERT INTO artifact_blobs(
                    artifact_id,
                    storage_kind,
                    blob_bytes,
                    storage_uri,
                    byte_size,
                    sha256,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    resolved_artifact_id,
                    resolved_storage_kind,
                    blob_bytes,
                    storage_uri,
                    len(blob_bytes) if blob_bytes is not None else 0,
                    hashlib.sha256(
                        blob_bytes
                        if blob_bytes is not None
                        else str(storage_uri or resolved_artifact_id).encode("utf-8")
                    ).hexdigest(),
                    timestamp,
                    timestamp,
                ),
            )
        if commit:
            await self._connection.commit()
        created = await self.get_artifact(resolved_artifact_id, user_id)
        if created is None:
            raise RuntimeError(f"Failed to create artifact {resolved_artifact_id}")
        return created

    async def get_artifact(self, artifact_id: str, user_id: str) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM artifacts
            WHERE id = ?
              AND user_id = ?
            """,
            (artifact_id, user_id),
        )

    async def list_artifacts(
        self,
        user_id: str,
        *,
        workspace_id: str | None = None,
        conversation_id: str | None = None,
        message_id: str | None = None,
        include_deleted: bool = False,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["user_id = ?"]
        parameters: list[Any] = [user_id]
        if workspace_id is not None:
            clauses.append("workspace_id = ?")
            parameters.append(workspace_id)
        if conversation_id is not None:
            clauses.append("conversation_id = ?")
            parameters.append(conversation_id)
        if message_id is not None:
            clauses.append("message_id = ?")
            parameters.append(message_id)
        if not include_deleted:
            clauses.append("status NOT IN ('deleted', 'purged')")
        query = f"""
            SELECT *
            FROM artifacts
            WHERE {" AND ".join(clauses)}
            ORDER BY updated_at DESC, id ASC
        """
        if limit is not None:
            query += " LIMIT ?"
            parameters.append(limit)
        return await self._fetch_all(query, tuple(parameters))

    async def delete_artifact(
        self,
        artifact_id: str,
        user_id: str,
        *,
        purge: bool = False,
        commit: bool = True,
    ) -> dict[str, Any] | None:
        existing = await self.get_artifact(artifact_id, user_id)
        if existing is None:
            return None
        timestamp = self._timestamp()
        if purge:
            await self._connection.execute(
                "DELETE FROM artifact_links WHERE artifact_id = ? AND user_id = ?",
                (artifact_id, user_id),
            )
            await self._connection.execute(
                "DELETE FROM artifact_chunks WHERE artifact_id = ? AND user_id = ?",
                (artifact_id, user_id),
            )
            await self._connection.execute(
                "DELETE FROM artifact_blobs WHERE artifact_id = ?",
                (artifact_id,),
            )
        await self._connection.execute(
            """
            UPDATE artifacts
            SET status = ?,
                updated_at = ?,
                deleted_at = ?
            WHERE id = ?
              AND user_id = ?
            """,
            ("purged" if purge else "deleted", timestamp, timestamp, artifact_id, user_id),
        )
        if commit:
            await self._connection.commit()
        return await self.get_artifact(artifact_id, user_id)

    async def create_artifact_chunk(
        self,
        *,
        artifact_id: str,
        user_id: str,
        chunk_index: int,
        text: str,
        token_count: int,
        kind: str,
        source_start_offset: int | None = None,
        source_end_offset: int | None = None,
        chunk_id: str | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        resolved_chunk_id = chunk_id or generate_prefixed_id("arc")
        timestamp = self._timestamp()
        await self._connection.execute(
            """
            INSERT INTO artifact_chunks(
                id,
                artifact_id,
                user_id,
                chunk_index,
                source_start_offset,
                source_end_offset,
                text,
                token_count,
                kind,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                resolved_chunk_id,
                artifact_id,
                user_id,
                chunk_index,
                source_start_offset,
                source_end_offset,
                text,
                token_count,
                kind,
                timestamp,
                timestamp,
            ),
        )
        if commit:
            await self._connection.commit()
        row = await self._fetch_one(
            "SELECT * FROM artifact_chunks WHERE id = ? AND user_id = ?",
            (resolved_chunk_id, user_id),
        )
        if row is None:
            raise RuntimeError(f"Failed to create artifact chunk {resolved_chunk_id}")
        return row

    async def create_artifact_chunks_bulk(
        self,
        chunks: Iterable[dict[str, Any]],
        *,
        commit: bool = True,
    ) -> list[dict[str, Any]]:
        created: list[dict[str, Any]] = []
        for chunk in chunks:
            created.append(
                await self.create_artifact_chunk(
                    artifact_id=str(chunk["artifact_id"]),
                    user_id=str(chunk["user_id"]),
                    chunk_index=int(chunk["chunk_index"]),
                    text=str(chunk["text"]),
                    token_count=int(chunk["token_count"]),
                    kind=str(chunk["kind"]),
                    source_start_offset=chunk.get("source_start_offset"),
                    source_end_offset=chunk.get("source_end_offset"),
                    chunk_id=str(chunk["id"]) if chunk.get("id") is not None else None,
                    commit=False,
                )
            )
        if commit:
            await self._connection.commit()
        return created

    async def list_artifact_chunks(self, artifact_id: str, user_id: str) -> list[dict[str, Any]]:
        return await self._fetch_all(
            """
            SELECT *
            FROM artifact_chunks
            WHERE artifact_id = ?
              AND user_id = ?
            ORDER BY chunk_index ASC, id ASC
            """,
            (artifact_id, user_id),
        )

    async def create_artifact_link(
        self,
        *,
        user_id: str,
        message_id: str,
        artifact_id: str,
        relation_kind: str = "attachment",
        ordinal: int = 0,
        link_id: str | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        resolved_link_id = link_id or generate_prefixed_id("arl")
        timestamp = self._timestamp()
        await self._connection.execute(
            """
            INSERT INTO artifact_links(
                id,
                user_id,
                message_id,
                artifact_id,
                relation_kind,
                ordinal,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                resolved_link_id,
                user_id,
                message_id,
                artifact_id,
                relation_kind,
                ordinal,
                timestamp,
            ),
        )
        if commit:
            await self._connection.commit()
        row = await self._fetch_one(
            "SELECT * FROM artifact_links WHERE id = ? AND user_id = ?",
            (resolved_link_id, user_id),
        )
        if row is None:
            raise RuntimeError(f"Failed to create artifact link {resolved_link_id}")
        return row

    async def create_artifact_links_bulk(
        self,
        links: Iterable[dict[str, Any]],
        *,
        commit: bool = True,
    ) -> list[dict[str, Any]]:
        created: list[dict[str, Any]] = []
        for link in links:
            created.append(
                await self.create_artifact_link(
                    user_id=str(link["user_id"]),
                    message_id=str(link["message_id"]),
                    artifact_id=str(link["artifact_id"]),
                    relation_kind=str(link.get("relation_kind", "attachment")),
                    ordinal=int(link.get("ordinal", 0)),
                    link_id=str(link["id"]) if link.get("id") is not None else None,
                    commit=False,
                )
            )
        if commit:
            await self._connection.commit()
        return created

    async def search_artifact_chunks(
        self,
        *,
        user_id: str,
        query: str,
        privacy_ceiling: int,
        scope_filter: list[MemoryScope],
        assistant_mode_id: str,
        workspace_id: str | None,
        conversation_id: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        scope_clauses, scope_parameters = self._scope_clauses(
            scope_filter,
            assistant_mode_id=assistant_mode_id,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
        )
        if not scope_clauses:
            return []
        query_text = """
            SELECT
                ac.*,
                a.workspace_id,
                a.conversation_id,
                a.message_id,
                a.artifact_type,
                a.source_kind,
                a.source_ref,
                a.mime_type,
                a.filename,
                a.title,
                a.content_hash,
                a.size_bytes,
                a.page_count,
                c.assistant_mode_id AS conversation_assistant_mode_id,
                a.status,
                a.privacy_level,
                a.preserve_verbatim,
                a.skip_raw_by_default,
                a.requires_explicit_request,
                a.metadata_json AS artifact_metadata_json,
                a.summary_text AS artifact_summary_text,
                a.index_text AS artifact_index_text,
                a.created_at AS artifact_created_at,
                a.updated_at AS artifact_updated_at,
                bm25(artifact_chunks_fts) AS rank
            FROM artifact_chunks_fts
            JOIN artifact_chunks AS ac ON ac._rowid = artifact_chunks_fts.rowid
            JOIN artifacts AS a ON a.id = ac.artifact_id
            LEFT JOIN conversations AS c ON c.id = a.conversation_id
            WHERE a.user_id = ?
              AND a.status = 'ready'
              AND a.deleted_at IS NULL
              AND a.privacy_level <= ?
              AND ({scope_clauses})
              AND artifact_chunks_fts MATCH ?
            ORDER BY rank ASC, a.updated_at DESC, ac.chunk_index ASC, ac.id ASC
            LIMIT ?
        """.format(scope_clauses=" OR ".join(scope_clauses))
        cursor = await self._connection.execute(
            query_text,
            (
                user_id,
                privacy_ceiling,
                *scope_parameters,
                query,
                limit,
            ),
        )
        rows = await cursor.fetchall()
        return [row for row in (_decode_json_columns(row) for row in rows) if row is not None]

    @staticmethod
    def _scope_clauses(
        scope_filter: list[MemoryScope],
        *,
        assistant_mode_id: str,
        workspace_id: str | None,
        conversation_id: str,
    ) -> tuple[list[str], list[Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        for scope in scope_filter:
            if scope is MemoryScope.GLOBAL_USER:
                clauses.append("(a.workspace_id IS NULL AND a.conversation_id IS NULL)")
            elif scope is MemoryScope.ASSISTANT_MODE:
                clauses.append("c.assistant_mode_id = ?")
                parameters.append(assistant_mode_id)
            elif scope is MemoryScope.WORKSPACE and workspace_id is not None:
                clauses.append("(c.assistant_mode_id = ? AND a.workspace_id = ?)")
                parameters.extend([assistant_mode_id, workspace_id])
            elif scope is MemoryScope.CONVERSATION:
                clauses.append("(c.assistant_mode_id = ? AND a.conversation_id = ?)")
                parameters.extend([assistant_mode_id, conversation_id])
            elif scope is MemoryScope.EPHEMERAL_SESSION:
                clauses.append("(c.assistant_mode_id = ? AND a.conversation_id = ?)")
                parameters.extend([assistant_mode_id, conversation_id])
        return clauses, parameters
