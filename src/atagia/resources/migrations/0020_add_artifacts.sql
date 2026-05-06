CREATE TABLE IF NOT EXISTS artifacts (
    _rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    workspace_id TEXT REFERENCES workspaces(id) ON DELETE SET NULL,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
    message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
    artifact_type TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    source_ref TEXT,
    mime_type TEXT,
    filename TEXT,
    title TEXT,
    content_hash TEXT,
    size_bytes INTEGER,
    page_count INTEGER,
    status TEXT NOT NULL DEFAULT 'queued',
    privacy_level INTEGER NOT NULL DEFAULT 0,
    preserve_verbatim INTEGER NOT NULL DEFAULT 0,
    skip_raw_by_default INTEGER NOT NULL DEFAULT 1,
    requires_explicit_request INTEGER NOT NULL DEFAULT 1,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    summary_text TEXT,
    index_text TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT,
    CHECK (artifact_type IN ('url', 'pdf', 'image', 'base64', 'file', 'pasted_text', 'other')),
    CHECK (source_kind IN ('host_embedded', 'upload', 'url', 'base64', 'pasted_text', 'external_ref')),
    CHECK (status IN ('queued', 'processing', 'ready', 'failed', 'deleted', 'purged')),
    CHECK (privacy_level BETWEEN 0 AND 3),
    CHECK (size_bytes IS NULL OR size_bytes >= 0),
    CHECK (page_count IS NULL OR page_count >= 0)
);

CREATE INDEX IF NOT EXISTS artifacts_user_status_scope_idx
    ON artifacts(
        user_id,
        status,
        privacy_level,
        workspace_id,
        conversation_id,
        message_id,
        artifact_type,
        source_kind,
        updated_at DESC,
        id ASC
    );

CREATE INDEX IF NOT EXISTS artifacts_user_conversation_idx
    ON artifacts(user_id, conversation_id, updated_at DESC, id ASC);

CREATE INDEX IF NOT EXISTS artifacts_user_message_idx
    ON artifacts(user_id, message_id, created_at DESC, id ASC);

CREATE TABLE IF NOT EXISTS artifact_blobs (
    artifact_id TEXT PRIMARY KEY REFERENCES artifacts(id) ON DELETE CASCADE,
    storage_kind TEXT NOT NULL,
    blob_bytes BLOB,
    storage_uri TEXT,
    byte_size INTEGER NOT NULL DEFAULT 0,
    sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (storage_kind IN ('sqlite_blob', 'external_ref')),
    CHECK (byte_size >= 0)
);

CREATE INDEX IF NOT EXISTS artifact_blobs_storage_idx
    ON artifact_blobs(storage_kind, storage_uri);

CREATE TABLE IF NOT EXISTS artifact_chunks (
    _rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    artifact_id TEXT NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    source_start_offset INTEGER,
    source_end_offset INTEGER,
    text TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    kind TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (chunk_index >= 0),
    CHECK (token_count >= 0),
    CHECK (kind IN ('ocr', 'extracted', 'parsed', 'transcript', 'summary'))
);

CREATE INDEX IF NOT EXISTS artifact_chunks_user_artifact_idx
    ON artifact_chunks(user_id, artifact_id, chunk_index ASC);

CREATE INDEX IF NOT EXISTS artifact_chunks_user_updated_idx
    ON artifact_chunks(user_id, updated_at DESC, id ASC);

CREATE TABLE IF NOT EXISTS artifact_links (
    id TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    artifact_id TEXT NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
    relation_kind TEXT NOT NULL,
    ordinal INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    CHECK (relation_kind IN ('attachment', 'inline_ref', 'citation', 'imported_source')),
    CHECK (ordinal >= 0)
);

CREATE INDEX IF NOT EXISTS artifact_links_user_message_idx
    ON artifact_links(user_id, message_id, ordinal ASC, id ASC);

CREATE INDEX IF NOT EXISTS artifact_links_user_artifact_idx
    ON artifact_links(user_id, artifact_id, ordinal ASC, id ASC);

DROP TRIGGER IF EXISTS artifact_chunks_fts_ai;
DROP TRIGGER IF EXISTS artifact_chunks_fts_ad;
DROP TRIGGER IF EXISTS artifact_chunks_fts_au;
DROP TABLE IF EXISTS artifact_chunks_fts;

CREATE VIRTUAL TABLE artifact_chunks_fts
USING fts5(
    text,
    content='artifact_chunks',
    content_rowid='_rowid',
    tokenize='unicode61'
);

CREATE TRIGGER artifact_chunks_fts_ai
AFTER INSERT ON artifact_chunks
BEGIN
    INSERT INTO artifact_chunks_fts(rowid, text)
    VALUES (new._rowid, new.text);
END;

CREATE TRIGGER artifact_chunks_fts_ad
AFTER DELETE ON artifact_chunks
BEGIN
    INSERT INTO artifact_chunks_fts(artifact_chunks_fts, rowid, text)
    VALUES ('delete', old._rowid, old.text);
END;

CREATE TRIGGER artifact_chunks_fts_au
AFTER UPDATE ON artifact_chunks
BEGIN
    INSERT INTO artifact_chunks_fts(artifact_chunks_fts, rowid, text)
    VALUES ('delete', old._rowid, old.text);
    INSERT INTO artifact_chunks_fts(rowid, text)
    VALUES (new._rowid, new.text);
END;

INSERT INTO artifact_chunks_fts(artifact_chunks_fts) VALUES('rebuild');
