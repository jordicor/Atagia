CREATE TABLE spaces (
    id TEXT NOT NULL,
    owner_user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    boundary_mode TEXT NOT NULL DEFAULT 'focus' CHECK (
        boundary_mode IN ('focus', 'severance', 'privacy_vault', 'tagged')
    ),
    display_name TEXT,
    source_kind TEXT NOT NULL DEFAULT 'explicit' CHECK (
        source_kind IN ('explicit', 'workspace_id')
    ),
    source_id TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (owner_user_id, id),
    UNIQUE (owner_user_id, source_kind, source_id)
);

CREATE INDEX idx_spaces_owner_boundary
    ON spaces(owner_user_id, boundary_mode, id);

ALTER TABLE conversations ADD COLUMN active_space_id TEXT;

CREATE INDEX idx_conversations_active_space
    ON conversations(user_id, active_space_id, status, updated_at DESC)
    WHERE active_space_id IS NOT NULL;

ALTER TABLE messages ADD COLUMN space_id TEXT;

CREATE INDEX idx_messages_space
    ON messages(space_id, conversation_id, seq)
    WHERE space_id IS NOT NULL;

ALTER TABLE memory_objects ADD COLUMN space_id TEXT;
ALTER TABLE memory_objects ADD COLUMN space_boundary_mode TEXT CHECK (
    space_boundary_mode IN ('focus', 'severance', 'privacy_vault', 'tagged')
    OR space_boundary_mode IS NULL
);

CREATE INDEX idx_memory_objects_space
    ON memory_objects(user_id, space_id, scope_canonical, status, updated_at DESC)
    WHERE space_id IS NOT NULL;

CREATE INDEX idx_memory_objects_space_boundary
    ON memory_objects(user_id, space_boundary_mode, status, updated_at DESC)
    WHERE space_id IS NOT NULL;
