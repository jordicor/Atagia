CREATE TABLE IF NOT EXISTS verbatim_pins (
    _rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    workspace_id TEXT REFERENCES workspaces(id) ON DELETE SET NULL,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
    assistant_mode_id TEXT REFERENCES assistant_modes(id) ON DELETE SET NULL,
    scope TEXT NOT NULL,
    target_kind TEXT NOT NULL,
    target_id TEXT NOT NULL,
    target_span_start INTEGER,
    target_span_end INTEGER,
    canonical_text TEXT NOT NULL,
    index_text TEXT NOT NULL,
    privacy_level INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'active',
    reason TEXT,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    expires_at TEXT,
    deleted_at TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}',
    CHECK (scope IN ('global_user', 'assistant_mode', 'workspace', 'conversation', 'ephemeral_session')),
    CHECK (target_kind IN ('message', 'memory_object', 'text_span')),
    CHECK (status IN ('active', 'archived', 'expired', 'deleted')),
    CHECK (privacy_level BETWEEN 0 AND 3),
    CHECK (target_span_start IS NULL OR target_span_start >= 0),
    CHECK (target_span_end IS NULL OR target_span_end >= 0),
    CHECK (
        target_span_start IS NULL
        OR target_span_end IS NULL
        OR target_span_end >= target_span_start
    )
);

CREATE INDEX IF NOT EXISTS verbatim_pins_user_status_scope_target_idx
    ON verbatim_pins(user_id, status, scope, assistant_mode_id, workspace_id, conversation_id, target_kind, target_id);

CREATE INDEX IF NOT EXISTS verbatim_pins_user_status_updated_idx
    ON verbatim_pins(user_id, status, updated_at DESC, created_at DESC, id ASC);

CREATE INDEX IF NOT EXISTS verbatim_pins_user_expires_idx
    ON verbatim_pins(user_id, expires_at);

DROP TRIGGER IF EXISTS verbatim_pins_fts_ai;
DROP TRIGGER IF EXISTS verbatim_pins_fts_ad;
DROP TRIGGER IF EXISTS verbatim_pins_fts_au;
DROP TABLE IF EXISTS verbatim_pins_fts;

CREATE VIRTUAL TABLE verbatim_pins_fts
USING fts5(
    index_text,
    content='verbatim_pins',
    content_rowid='_rowid',
    tokenize='unicode61'
);

CREATE TRIGGER verbatim_pins_fts_ai
AFTER INSERT ON verbatim_pins
BEGIN
    INSERT INTO verbatim_pins_fts(rowid, index_text)
    VALUES (new._rowid, new.index_text);
END;

CREATE TRIGGER verbatim_pins_fts_ad
AFTER DELETE ON verbatim_pins
BEGIN
    INSERT INTO verbatim_pins_fts(verbatim_pins_fts, rowid, index_text)
    VALUES ('delete', old._rowid, old.index_text);
END;

CREATE TRIGGER verbatim_pins_fts_au
AFTER UPDATE ON verbatim_pins
BEGIN
    INSERT INTO verbatim_pins_fts(verbatim_pins_fts, rowid, index_text)
    VALUES ('delete', old._rowid, old.index_text);
    INSERT INTO verbatim_pins_fts(rowid, index_text)
    VALUES (new._rowid, new.index_text);
END;

INSERT INTO verbatim_pins_fts(verbatim_pins_fts) VALUES('rebuild');
