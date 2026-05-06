ALTER TABLE conversations ADD COLUMN temporary INTEGER NOT NULL DEFAULT 0 CHECK (temporary IN (0, 1));
ALTER TABLE conversations ADD COLUMN temporary_ttl_seconds INTEGER CHECK (temporary_ttl_seconds IS NULL OR temporary_ttl_seconds > 0);
ALTER TABLE conversations ADD COLUMN purge_on_close INTEGER NOT NULL DEFAULT 0 CHECK (purge_on_close IN (0, 1));
ALTER TABLE conversations ADD COLUMN last_activity_at TEXT;
ALTER TABLE conversations ADD COLUMN closed_at TEXT;

-- SQLite cannot add a status CHECK through ALTER TABLE. Conversation status is
-- validated by the ConversationStatus enum until a future table rebuild.

UPDATE conversations
SET last_activity_at = COALESCE(
    (
        SELECT MAX(COALESCE(m.occurred_at, m.created_at))
        FROM messages AS m
        WHERE m.conversation_id = conversations.id
    ),
    conversations.created_at
)
WHERE last_activity_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_conversations_user_status_temporary
    ON conversations(user_id, status, temporary, updated_at DESC, id ASC);

CREATE INDEX IF NOT EXISTS idx_conversations_temporary_ttl
    ON conversations(temporary, status, last_activity_at)
    WHERE temporary = 1 AND temporary_ttl_seconds IS NOT NULL;

ALTER TABLE memory_objects ADD COLUMN archived_by_conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_mo_archived_by_conversation
    ON memory_objects(user_id, archived_by_conversation_id, status);

CREATE TABLE IF NOT EXISTS memory_edit_history (
    _rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL REFERENCES memory_objects(id) ON DELETE CASCADE,
    previous_text TEXT NOT NULL,
    new_text TEXT NOT NULL,
    edited_by TEXT NOT NULL,
    edit_source TEXT NOT NULL CHECK (edit_source IN ('api', 'mcp', 'agent', 'admin')),
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_edit_history_memory
    ON memory_edit_history(memory_id);

CREATE TABLE IF NOT EXISTS deletion_tombstones (
    id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL CHECK (entity_type IN ('conversation', 'memory', 'user', 'workspace')),
    deleted_at TEXT NOT NULL,
    deletion_reason TEXT NOT NULL CHECK (
        deletion_reason IN ('user_request', 'ttl_expiry', 'right_to_erasure', 'admin', 'purge_on_close')
    ),
    deleted_by TEXT NOT NULL CHECK (deleted_by IN ('system', 'admin')),
    scope_summary TEXT
);

CREATE INDEX IF NOT EXISTS idx_tombstones_deleted_at
    ON deletion_tombstones(deleted_at);

CREATE TABLE IF NOT EXISTS pending_file_deletions (
    id TEXT PRIMARY KEY,
    storage_uri TEXT NOT NULL,
    storage_root TEXT NOT NULL,
    sha256 TEXT,
    reason TEXT NOT NULL CHECK (reason IN ('conversation_delete', 'user_erasure')),
    tombstone_id TEXT,
    created_at TEXT NOT NULL,
    attempted_at TEXT,
    deleted_at TEXT,
    last_error TEXT
);

CREATE INDEX IF NOT EXISTS idx_pending_file_deletions_open
    ON pending_file_deletions(deleted_at, created_at);
