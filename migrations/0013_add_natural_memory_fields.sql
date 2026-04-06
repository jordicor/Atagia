-- atagia:foreign_keys_off
BEGIN;

DROP TRIGGER IF EXISTS memory_objects_fts_ai;
DROP TRIGGER IF EXISTS memory_objects_fts_ad;
DROP TRIGGER IF EXISTS memory_objects_fts_au;
DROP TABLE IF EXISTS memory_objects_fts;

CREATE TABLE memory_objects_new (
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
    index_text TEXT,
    extraction_hash TEXT,
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
    memory_category TEXT NOT NULL DEFAULT 'unknown' CHECK (
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
    preserve_verbatim INTEGER NOT NULL DEFAULT 0 CHECK (preserve_verbatim IN (0, 1)),
    valid_from TEXT,
    valid_to TEXT,
    temporal_type TEXT NOT NULL DEFAULT 'unknown',
    tension_score REAL NOT NULL DEFAULT 0.0,
    tension_updated_at TEXT,
    status TEXT NOT NULL DEFAULT 'active' CHECK (
        status IN (
            'active',
            'superseded',
            'archived',
            'deleted',
            'review_required',
            'pending_user_confirmation',
            'declined'
        )
    ),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

INSERT INTO memory_objects_new(
    _rowid,
    id,
    user_id,
    workspace_id,
    conversation_id,
    assistant_mode_id,
    object_type,
    scope,
    canonical_text,
    index_text,
    extraction_hash,
    payload_json,
    source_kind,
    confidence,
    stability,
    vitality,
    maya_score,
    privacy_level,
    memory_category,
    preserve_verbatim,
    valid_from,
    valid_to,
    temporal_type,
    tension_score,
    tension_updated_at,
    status,
    created_at,
    updated_at
)
SELECT
    _rowid,
    id,
    user_id,
    workspace_id,
    conversation_id,
    assistant_mode_id,
    object_type,
    scope,
    canonical_text,
    index_text,
    extraction_hash,
    payload_json,
    source_kind,
    confidence,
    stability,
    vitality,
    maya_score,
    privacy_level,
    'unknown',
    0,
    valid_from,
    valid_to,
    temporal_type,
    tension_score,
    tension_updated_at,
    status,
    created_at,
    updated_at
FROM memory_objects;

DROP TABLE memory_objects;
ALTER TABLE memory_objects_new RENAME TO memory_objects;

CREATE INDEX idx_memory_objects_user_type_scope
    ON memory_objects(user_id, object_type, scope);

CREATE INDEX idx_memory_objects_workspace
    ON memory_objects(workspace_id);

CREATE INDEX idx_memory_objects_mode
    ON memory_objects(assistant_mode_id);

CREATE INDEX idx_memory_objects_status
    ON memory_objects(status);

CREATE INDEX idx_memory_objects_user_status
    ON memory_objects(user_id, status);

CREATE UNIQUE INDEX uq_memory_objects_user_extraction_hash
    ON memory_objects(user_id, extraction_hash)
    WHERE extraction_hash IS NOT NULL;

CREATE INDEX idx_mo_tension
    ON memory_objects(user_id, tension_score)
    WHERE tension_score > 0 AND object_type = 'belief';

CREATE INDEX idx_mo_temporal_expiry
    ON memory_objects(valid_to, status, object_type)
    WHERE valid_to IS NOT NULL AND status = 'active';

CREATE VIRTUAL TABLE memory_objects_fts
USING fts5(
    canonical_text,
    index_text,
    content='memory_objects',
    content_rowid='_rowid',
    tokenize='unicode61'
);

CREATE TRIGGER memory_objects_fts_ai
AFTER INSERT ON memory_objects
BEGIN
    INSERT INTO memory_objects_fts(rowid, canonical_text, index_text)
    VALUES (new._rowid, new.canonical_text, new.index_text);
END;

CREATE TRIGGER memory_objects_fts_ad
AFTER DELETE ON memory_objects
BEGIN
    INSERT INTO memory_objects_fts(memory_objects_fts, rowid, canonical_text, index_text)
    VALUES ('delete', old._rowid, old.canonical_text, old.index_text);
END;

CREATE TRIGGER memory_objects_fts_au
AFTER UPDATE ON memory_objects
BEGIN
    INSERT INTO memory_objects_fts(memory_objects_fts, rowid, canonical_text, index_text)
    VALUES ('delete', old._rowid, old.canonical_text, old.index_text);
    INSERT INTO memory_objects_fts(rowid, canonical_text, index_text)
    VALUES (new._rowid, new.canonical_text, new.index_text);
END;

INSERT INTO memory_objects_fts(memory_objects_fts) VALUES('rebuild');

CREATE TABLE memory_consent_profile (
    _rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    category TEXT NOT NULL CHECK (
        category IN (
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
    confirmed_count INTEGER NOT NULL DEFAULT 0,
    declined_count INTEGER NOT NULL DEFAULT 0,
    last_confirmed_at TEXT,
    last_declined_at TEXT,
    updated_at TEXT NOT NULL,
    UNIQUE(user_id, category)
);

CREATE INDEX idx_mcp_user_category
    ON memory_consent_profile(user_id, category);

COMMIT;
