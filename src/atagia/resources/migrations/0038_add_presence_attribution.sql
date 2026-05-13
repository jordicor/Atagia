CREATE TABLE presences (
    id TEXT NOT NULL,
    owner_user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    kind TEXT NOT NULL DEFAULT 'unknown' CHECK (
        kind IN ('human', 'owned_ai', 'owned_facet', 'external_actor', 'overseer', 'unknown')
    ),
    display_name TEXT,
    source_kind TEXT NOT NULL DEFAULT 'explicit' CHECK (
        source_kind IN ('explicit', 'character_id', 'default_ai', 'human_owner')
    ),
    source_id TEXT NOT NULL,
    presence_cluster_id TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (owner_user_id, id),
    UNIQUE (owner_user_id, source_kind, source_id)
);

CREATE INDEX idx_presences_owner_kind
    ON presences(owner_user_id, kind, id);

CREATE INDEX idx_presences_owner_cluster
    ON presences(owner_user_id, presence_cluster_id, id)
    WHERE presence_cluster_id IS NOT NULL;

ALTER TABLE conversations ADD COLUMN active_presence_id TEXT;

CREATE INDEX idx_conversations_active_presence
    ON conversations(user_id, active_presence_id, status, updated_at DESC)
    WHERE active_presence_id IS NOT NULL;

ALTER TABLE messages ADD COLUMN active_presence_id TEXT;
ALTER TABLE messages ADD COLUMN source_presence_id TEXT;

CREATE INDEX idx_messages_active_presence
    ON messages(active_presence_id, conversation_id, seq)
    WHERE active_presence_id IS NOT NULL;

CREATE INDEX idx_messages_source_presence
    ON messages(source_presence_id, conversation_id, seq)
    WHERE source_presence_id IS NOT NULL;

ALTER TABLE memory_objects ADD COLUMN active_presence_id TEXT;
ALTER TABLE memory_objects ADD COLUMN source_presence_id TEXT;
ALTER TABLE memory_objects ADD COLUMN presence_cluster_id TEXT;

CREATE INDEX idx_memory_objects_active_presence
    ON memory_objects(user_id, active_presence_id, scope_canonical, status, updated_at DESC)
    WHERE active_presence_id IS NOT NULL;

CREATE INDEX idx_memory_objects_source_presence
    ON memory_objects(user_id, source_presence_id, scope_canonical, status, updated_at DESC)
    WHERE source_presence_id IS NOT NULL;

CREATE INDEX idx_memory_objects_presence_cluster
    ON memory_objects(user_id, presence_cluster_id, scope_canonical, status, updated_at DESC)
    WHERE presence_cluster_id IS NOT NULL;

CREATE TABLE memory_object_subjects (
    memory_object_id TEXT NOT NULL REFERENCES memory_objects(id) ON DELETE CASCADE,
    owner_user_id TEXT NOT NULL,
    subject_presence_id TEXT NOT NULL,
    relation TEXT NOT NULL DEFAULT 'subject',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (memory_object_id, subject_presence_id, relation),
    FOREIGN KEY (owner_user_id, subject_presence_id)
        REFERENCES presences(owner_user_id, id)
        ON DELETE CASCADE
);

CREATE INDEX idx_memory_object_subjects_owner_presence
    ON memory_object_subjects(owner_user_id, subject_presence_id, relation, memory_object_id);
