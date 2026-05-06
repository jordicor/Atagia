CREATE TABLE pending_memory_confirmations (
    _rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    memory_id TEXT NOT NULL REFERENCES memory_objects(id) ON DELETE CASCADE,
    memory_category TEXT NOT NULL CHECK (
        memory_category IN (
            'phone',
            'address',
            'pin_or_password',
            'medication',
            'financial',
            'date_of_birth',
            'contact_identity',
            'other_sensitive',
            'unknown'
        )
    ),
    created_at TEXT NOT NULL,
    asked_at TEXT,
    confirmation_asked_once INTEGER NOT NULL DEFAULT 0 CHECK (confirmation_asked_once IN (0, 1)),
    UNIQUE(user_id, memory_id)
);

CREATE INDEX idx_pmc_user_conversation_created
    ON pending_memory_confirmations(user_id, conversation_id, created_at, _rowid);

CREATE INDEX idx_pmc_user_conversation_category_asked
    ON pending_memory_confirmations(
        user_id,
        conversation_id,
        memory_category,
        asked_at,
        created_at,
        _rowid
    );
