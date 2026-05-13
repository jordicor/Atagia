CREATE TABLE IF NOT EXISTS artifact_payload_blobs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    storage_kind TEXT NOT NULL,
    identity_kind TEXT NOT NULL,
    content_sha256 TEXT,
    byte_size INTEGER NOT NULL DEFAULT 0,
    blob_bytes BLOB,
    storage_key TEXT,
    external_uri TEXT,
    status TEXT NOT NULL DEFAULT 'ready',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (storage_kind IN ('sqlite_blob', 'external_ref', 'local_file')),
    CHECK (identity_kind IN ('content_sha256', 'external_uri')),
    CHECK (status IN ('pending', 'ready', 'gc_pending', 'quarantined', 'deleted')),
    CHECK (byte_size >= 0),
    CHECK (
        (storage_kind = 'sqlite_blob' AND blob_bytes IS NOT NULL AND storage_key IS NULL AND external_uri IS NULL)
        OR (storage_kind = 'local_file' AND blob_bytes IS NULL AND storage_key IS NOT NULL AND external_uri IS NULL)
        OR (storage_kind = 'external_ref' AND blob_bytes IS NULL AND storage_key IS NULL AND external_uri IS NOT NULL)
    ),
    CHECK (
        (identity_kind = 'content_sha256' AND content_sha256 IS NOT NULL)
        OR (identity_kind = 'external_uri' AND external_uri IS NOT NULL)
    )
);

ALTER TABLE artifacts
    ADD COLUMN payload_blob_id TEXT REFERENCES artifact_payload_blobs(id) ON DELETE SET NULL;

CREATE UNIQUE INDEX IF NOT EXISTS artifact_payload_blobs_active_content_identity_idx
    ON artifact_payload_blobs(user_id, storage_kind, content_sha256, byte_size)
    WHERE identity_kind = 'content_sha256'
      AND status IN ('pending', 'ready', 'gc_pending');

CREATE UNIQUE INDEX IF NOT EXISTS artifact_payload_blobs_active_external_identity_idx
    ON artifact_payload_blobs(user_id, external_uri)
    WHERE identity_kind = 'external_uri'
      AND status IN ('pending', 'ready', 'gc_pending');

CREATE UNIQUE INDEX IF NOT EXISTS artifact_payload_blobs_active_storage_key_idx
    ON artifact_payload_blobs(storage_key)
    WHERE storage_kind = 'local_file'
      AND status IN ('pending', 'ready', 'gc_pending');

CREATE INDEX IF NOT EXISTS artifact_payload_blobs_user_status_idx
    ON artifact_payload_blobs(user_id, status, updated_at DESC, id ASC);

CREATE INDEX IF NOT EXISTS artifacts_user_payload_blob_idx
    ON artifacts(user_id, payload_blob_id, status, id ASC);
