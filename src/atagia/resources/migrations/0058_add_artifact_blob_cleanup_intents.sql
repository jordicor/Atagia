CREATE TABLE IF NOT EXISTS artifact_blob_cleanup_intents (
    storage_identity TEXT PRIMARY KEY,
    storage_root TEXT NOT NULL,
    storage_uri TEXT NOT NULL,
    expected_sha256 TEXT,
    expected_byte_size INTEGER,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    verified_at TEXT,
    last_attempt_at TEXT,
    last_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (expected_byte_size IS NULL OR expected_byte_size >= 0),
    CHECK (attempt_count >= 0)
);

CREATE INDEX IF NOT EXISTS artifact_blob_cleanup_intents_attempt_idx
    ON artifact_blob_cleanup_intents(last_attempt_at, created_at, storage_identity);
