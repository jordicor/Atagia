-- Durable negative identities for explicitly retired extracted memories.

CREATE TABLE memory_extraction_suppressions (
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    source_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    identity_hash TEXT NOT NULL CHECK (TRIM(identity_hash) <> ''),
    extraction_hash TEXT CHECK (
        extraction_hash IS NULL OR TRIM(extraction_hash) <> ''
    ),
    retired_memory_id TEXT NOT NULL CHECK (TRIM(retired_memory_id) <> ''),
    replacement_memory_id TEXT,
    reason TEXT NOT NULL CHECK (
        reason IN (
            'memory_edited',
            'memory_archived',
            'memory_hard_deleted',
            'conversation_archived',
            'conversation_deleted',
            'conversation_purged'
        )
    ),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, source_message_id, retired_memory_id)
) WITHOUT ROWID;
