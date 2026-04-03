CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    external_ref TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS assistant_modes (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    prompt_hash TEXT,
    memory_policy_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS workspaces (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    name TEXT NOT NULL,
    metadata_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_workspaces_user
    ON workspaces(user_id);

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    workspace_id TEXT REFERENCES workspaces(id) ON DELETE SET NULL,
    assistant_mode_id TEXT NOT NULL REFERENCES assistant_modes(id) ON DELETE RESTRICT,
    title TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    metadata_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_conversations_user
    ON conversations(user_id);

CREATE INDEX IF NOT EXISTS idx_conversations_workspace
    ON conversations(workspace_id);

CREATE TABLE IF NOT EXISTS messages (
    _rowid INTEGER PRIMARY KEY,
    id TEXT NOT NULL UNIQUE,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
    seq INTEGER NOT NULL,
    text TEXT NOT NULL,
    token_count INTEGER,
    metadata_json TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(conversation_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation_seq
    ON messages(conversation_id, seq);

CREATE TABLE IF NOT EXISTS memory_objects (
    _rowid INTEGER PRIMARY KEY,
    id TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    workspace_id TEXT REFERENCES workspaces(id) ON DELETE SET NULL,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
    assistant_mode_id TEXT REFERENCES assistant_modes(id) ON DELETE SET NULL,
    object_type TEXT NOT NULL CHECK (
        object_type IN (
            'evidence',
            'belief',
            'interaction_contract',
            'state_snapshot',
            'consequence_chain',
            'summary_view'
        )
    ),
    scope TEXT NOT NULL CHECK (
        scope IN (
            'global_user',
            'assistant_mode',
            'workspace',
            'conversation',
            'ephemeral_session'
        )
    ),
    canonical_text TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    source_kind TEXT NOT NULL CHECK (
        source_kind IN (
            'verbatim',
            'extracted',
            'inferred',
            'summarized',
            'composed'
        )
    ),
    confidence REAL NOT NULL DEFAULT 0.5,
    stability REAL NOT NULL DEFAULT 0.5,
    vitality REAL NOT NULL DEFAULT 0.0,
    maya_score REAL NOT NULL DEFAULT 0.0,
    privacy_level INTEGER NOT NULL DEFAULT 0,
    valid_from TEXT,
    valid_to TEXT,
    status TEXT NOT NULL DEFAULT 'active' CHECK (
        status IN ('active', 'superseded', 'archived', 'deleted', 'review_required')
    ),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_objects_user_type_scope
    ON memory_objects(user_id, object_type, scope);

CREATE INDEX IF NOT EXISTS idx_memory_objects_workspace
    ON memory_objects(workspace_id);

CREATE INDEX IF NOT EXISTS idx_memory_objects_mode
    ON memory_objects(assistant_mode_id);

CREATE INDEX IF NOT EXISTS idx_memory_objects_status
    ON memory_objects(status);

CREATE INDEX IF NOT EXISTS idx_memory_objects_user_status
    ON memory_objects(user_id, status);

CREATE TABLE IF NOT EXISTS belief_versions (
    belief_id TEXT NOT NULL REFERENCES memory_objects(id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    claim_key TEXT NOT NULL,
    claim_value_json TEXT NOT NULL,
    condition_json TEXT NOT NULL DEFAULT '{}',
    support_count INTEGER NOT NULL DEFAULT 0,
    contradict_count INTEGER NOT NULL DEFAULT 0,
    supersedes_version INTEGER,
    is_current INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    PRIMARY KEY (belief_id, version)
);

CREATE INDEX IF NOT EXISTS idx_belief_versions_current
    ON belief_versions(belief_id, is_current);

CREATE INDEX IF NOT EXISTS idx_belief_versions_claim
    ON belief_versions(claim_key);

CREATE TABLE IF NOT EXISTS memory_links (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    src_memory_id TEXT NOT NULL REFERENCES memory_objects(id) ON DELETE CASCADE,
    dst_memory_id TEXT NOT NULL REFERENCES memory_objects(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL CHECK (
        relation_type IN (
            'supports',
            'contradicts',
            'depends_on',
            'derived_from',
            'supersedes',
            'exception_to',
            'about_topic',
            'mentions_entity',
            'applies_in_mode',
            'belongs_to_workspace',
            'led_to',
            'reinforces',
            'weakens'
        )
    ),
    weight REAL NOT NULL DEFAULT 1.0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_links_user_src
    ON memory_links(user_id, src_memory_id);

CREATE INDEX IF NOT EXISTS idx_memory_links_dst
    ON memory_links(dst_memory_id);

CREATE INDEX IF NOT EXISTS idx_memory_links_relation
    ON memory_links(relation_type);

CREATE TABLE IF NOT EXISTS retrieval_events (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    request_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    response_message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
    assistant_mode_id TEXT NOT NULL REFERENCES assistant_modes(id) ON DELETE RESTRICT,
    retrieval_plan_json TEXT NOT NULL,
    selected_memory_ids_json TEXT NOT NULL,
    context_view_json TEXT NOT NULL,
    outcome_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_retrieval_events_user
    ON retrieval_events(user_id, created_at);

CREATE INDEX IF NOT EXISTS idx_retrieval_events_conversation
    ON retrieval_events(conversation_id);

CREATE TABLE IF NOT EXISTS memory_feedback_events (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    retrieval_event_id TEXT REFERENCES retrieval_events(id) ON DELETE CASCADE,
    memory_id TEXT REFERENCES memory_objects(id) ON DELETE CASCADE,
    feedback_type TEXT NOT NULL CHECK (
        feedback_type IN (
            'used',
            'useful',
            'irrelevant',
            'intrusive',
            'stale',
            'wrong_scope',
            'corrected_by_user',
            'confirmed_by_user'
        )
    ),
    score REAL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_feedback_memory
    ON memory_feedback_events(memory_id, feedback_type);

CREATE INDEX IF NOT EXISTS idx_feedback_user
    ON memory_feedback_events(user_id);

CREATE INDEX IF NOT EXISTS idx_feedback_retrieval
    ON memory_feedback_events(retrieval_event_id);

CREATE TABLE IF NOT EXISTS contract_dimensions_current (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    workspace_id TEXT REFERENCES workspaces(id) ON DELETE SET NULL,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
    assistant_mode_id TEXT REFERENCES assistant_modes(id) ON DELETE SET NULL,
    scope TEXT NOT NULL CHECK (
        scope IN (
            'global_user',
            'assistant_mode',
            'workspace',
            'conversation',
            'ephemeral_session'
        )
    ),
    dimension_name TEXT NOT NULL,
    value_json TEXT NOT NULL,
    confidence REAL NOT NULL,
    source_memory_id TEXT NOT NULL REFERENCES memory_objects(id) ON DELETE CASCADE,
    updated_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_contract_dimensions_current
    ON contract_dimensions_current(
        user_id,
        COALESCE(workspace_id, ''),
        COALESCE(conversation_id, ''),
        COALESCE(assistant_mode_id, ''),
        scope,
        dimension_name
    );

CREATE INDEX IF NOT EXISTS idx_contract_dims_user
    ON contract_dimensions_current(user_id);

CREATE TABLE IF NOT EXISTS consequence_chains (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    workspace_id TEXT REFERENCES workspaces(id) ON DELETE SET NULL,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
    assistant_mode_id TEXT REFERENCES assistant_modes(id) ON DELETE SET NULL,
    action_memory_id TEXT NOT NULL REFERENCES memory_objects(id) ON DELETE CASCADE,
    outcome_memory_id TEXT NOT NULL REFERENCES memory_objects(id) ON DELETE CASCADE,
    tendency_belief_id TEXT REFERENCES memory_objects(id) ON DELETE SET NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_consequence_chains_user
    ON consequence_chains(user_id);

CREATE INDEX IF NOT EXISTS idx_consequence_chains_workspace
    ON consequence_chains(workspace_id);

CREATE TABLE IF NOT EXISTS summary_views (
    id TEXT PRIMARY KEY,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
    workspace_id TEXT REFERENCES workspaces(id) ON DELETE SET NULL,
    source_message_start_seq INTEGER NOT NULL,
    source_message_end_seq INTEGER NOT NULL,
    summary_kind TEXT NOT NULL CHECK (
        summary_kind IN ('conversation_chunk', 'workspace_rollup', 'context_view')
    ),
    summary_text TEXT NOT NULL,
    source_object_ids_json TEXT NOT NULL DEFAULT '[]',
    maya_score REAL NOT NULL DEFAULT 1.5,
    created_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
USING fts5(
    text,
    content='messages',
    content_rowid='_rowid',
    tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS messages_fts_ai
AFTER INSERT ON messages
BEGIN
    INSERT INTO messages_fts(rowid, text)
    VALUES (new._rowid, new.text);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_ad
AFTER DELETE ON messages
BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, text)
    VALUES ('delete', old._rowid, old.text);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_au
AFTER UPDATE ON messages
BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, text)
    VALUES ('delete', old._rowid, old.text);
    INSERT INTO messages_fts(rowid, text)
    VALUES (new._rowid, new.text);
END;

CREATE VIRTUAL TABLE IF NOT EXISTS memory_objects_fts
USING fts5(
    canonical_text,
    content='memory_objects',
    content_rowid='_rowid',
    tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS memory_objects_fts_ai
AFTER INSERT ON memory_objects
BEGIN
    INSERT INTO memory_objects_fts(rowid, canonical_text)
    VALUES (new._rowid, new.canonical_text);
END;

CREATE TRIGGER IF NOT EXISTS memory_objects_fts_ad
AFTER DELETE ON memory_objects
BEGIN
    INSERT INTO memory_objects_fts(memory_objects_fts, rowid, canonical_text)
    VALUES ('delete', old._rowid, old.canonical_text);
END;

CREATE TRIGGER IF NOT EXISTS memory_objects_fts_au
AFTER UPDATE ON memory_objects
BEGIN
    INSERT INTO memory_objects_fts(memory_objects_fts, rowid, canonical_text)
    VALUES ('delete', old._rowid, old.canonical_text);
    INSERT INTO memory_objects_fts(rowid, canonical_text)
    VALUES (new._rowid, new.canonical_text);
END;
