CREATE TABLE IF NOT EXISTS initial_context_packages (
    _rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    package_key_hash TEXT NOT NULL UNIQUE,
    package_kind TEXT NOT NULL CHECK (package_kind IN ('baseline', 'conversation')),
    version INTEGER NOT NULL CHECK (version > 0),
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
    retrieval_profile_id TEXT NOT NULL,
    key_json TEXT NOT NULL DEFAULT '{}',
    policy_signature_json TEXT NOT NULL DEFAULT '{}',
    coordinate_signature_json TEXT NOT NULL DEFAULT '{}',
    source_fingerprint_json TEXT NOT NULL DEFAULT '{}',
    blocks_json TEXT NOT NULL DEFAULT '{}',
    source_refs_json TEXT NOT NULL DEFAULT '{}',
    diagnostics_json TEXT NOT NULL DEFAULT '{}',
    build_status TEXT NOT NULL DEFAULT 'active' CHECK (
        build_status IN ('active', 'stale', 'failed', 'deleted')
    ),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    valid_until TEXT,
    CHECK (package_kind = 'conversation' OR conversation_id IS NULL),
    CHECK (package_kind <> 'conversation' OR conversation_id IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS idx_initial_context_packages_user_kind
    ON initial_context_packages(user_id, package_kind, build_status, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_initial_context_packages_user_conversation
    ON initial_context_packages(user_id, conversation_id, build_status, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_initial_context_packages_user_profile_kind
    ON initial_context_packages(user_id, retrieval_profile_id, package_kind);

CREATE TRIGGER IF NOT EXISTS initial_context_packages_conversation_owner_bi
BEFORE INSERT ON initial_context_packages
BEGIN
    SELECT RAISE(ABORT, 'initial_context_packages conversation_id must belong to user_id')
    WHERE new.conversation_id IS NOT NULL
      AND NOT EXISTS (
          SELECT 1
          FROM conversations
          WHERE conversations.id = new.conversation_id
            AND conversations.user_id = new.user_id
      );
END;

CREATE TRIGGER IF NOT EXISTS initial_context_packages_conversation_owner_bu
BEFORE UPDATE OF user_id, conversation_id ON initial_context_packages
BEGIN
    SELECT RAISE(ABORT, 'initial_context_packages conversation_id must belong to user_id')
    WHERE new.conversation_id IS NOT NULL
      AND NOT EXISTS (
          SELECT 1
          FROM conversations
          WHERE conversations.id = new.conversation_id
            AND conversations.user_id = new.user_id
      );
END;
