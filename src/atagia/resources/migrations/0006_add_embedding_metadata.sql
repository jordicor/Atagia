CREATE TABLE IF NOT EXISTS memory_embedding_metadata (
    memory_id TEXT PRIMARY KEY REFERENCES memory_objects(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,
    object_type TEXT NOT NULL,
    scope TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_embedding_metadata_user
    ON memory_embedding_metadata(user_id);
