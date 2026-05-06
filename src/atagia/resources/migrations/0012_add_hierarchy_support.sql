ALTER TABLE summary_views ADD COLUMN user_id TEXT;
ALTER TABLE summary_views ADD COLUMN hierarchy_level INTEGER NOT NULL DEFAULT 0;

UPDATE summary_views
SET user_id = (
    SELECT c.user_id
    FROM conversations AS c
    WHERE c.id = summary_views.conversation_id
)
WHERE conversation_id IS NOT NULL;

UPDATE summary_views
SET user_id = (
    SELECT w.user_id
    FROM workspaces AS w
    WHERE w.id = summary_views.workspace_id
)
WHERE user_id IS NULL
  AND workspace_id IS NOT NULL;

DELETE FROM summary_views
WHERE user_id IS NULL;

CREATE TABLE summary_views_new (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
    workspace_id TEXT REFERENCES workspaces(id) ON DELETE SET NULL,
    source_message_start_seq INTEGER,
    source_message_end_seq INTEGER,
    summary_kind TEXT NOT NULL CHECK (
        summary_kind IN (
            'conversation_chunk',
            'workspace_rollup',
            'context_view',
            'episode',
            'thematic_profile'
        )
    ),
    summary_text TEXT NOT NULL,
    source_object_ids_json TEXT NOT NULL DEFAULT '[]',
    maya_score REAL NOT NULL DEFAULT 1.5,
    model TEXT NOT NULL DEFAULT '',
    hierarchy_level INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

INSERT INTO summary_views_new (
    id,
    user_id,
    conversation_id,
    workspace_id,
    source_message_start_seq,
    source_message_end_seq,
    summary_kind,
    summary_text,
    source_object_ids_json,
    maya_score,
    model,
    hierarchy_level,
    created_at
)
SELECT
    id,
    user_id,
    conversation_id,
    workspace_id,
    source_message_start_seq,
    source_message_end_seq,
    summary_kind,
    summary_text,
    source_object_ids_json,
    maya_score,
    model,
    hierarchy_level,
    created_at
FROM summary_views;

DROP TABLE summary_views;
ALTER TABLE summary_views_new RENAME TO summary_views;

CREATE INDEX IF NOT EXISTS idx_sv_user_kind_level
    ON summary_views(user_id, summary_kind, hierarchy_level, created_at);
