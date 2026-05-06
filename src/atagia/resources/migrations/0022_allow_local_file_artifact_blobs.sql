-- atagia:foreign_keys_off
BEGIN;

DROP INDEX IF EXISTS artifact_blobs_storage_idx;

CREATE TABLE artifact_blobs_new (
    artifact_id TEXT PRIMARY KEY REFERENCES artifacts(id) ON DELETE CASCADE,
    storage_kind TEXT NOT NULL,
    blob_bytes BLOB,
    storage_uri TEXT,
    byte_size INTEGER NOT NULL DEFAULT 0,
    sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (storage_kind IN ('sqlite_blob', 'external_ref', 'local_file')),
    CHECK (byte_size >= 0)
);

INSERT INTO artifact_blobs_new(
    artifact_id,
    storage_kind,
    blob_bytes,
    storage_uri,
    byte_size,
    sha256,
    created_at,
    updated_at
)
SELECT
    artifact_id,
    storage_kind,
    blob_bytes,
    storage_uri,
    byte_size,
    sha256,
    created_at,
    updated_at
FROM artifact_blobs;

DROP TABLE artifact_blobs;
ALTER TABLE artifact_blobs_new RENAME TO artifact_blobs;

CREATE INDEX IF NOT EXISTS artifact_blobs_storage_idx
    ON artifact_blobs(storage_kind, storage_uri);

COMMIT;
