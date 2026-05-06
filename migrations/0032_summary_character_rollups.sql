-- Phase 8: make character rollups a first-class summary kind.
--
-- SQLite cannot widen a CHECK constraint in place, so rebuild summary_views
-- with the same post-0031 columns and include character_rollup.

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
            'character_rollup',
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
    created_at TEXT NOT NULL,
    intimacy_boundary TEXT NOT NULL DEFAULT 'ordinary' CHECK (
        intimacy_boundary IN (
            'ordinary',
            'romantic_private',
            'intimacy_private',
            'intimacy_preference_private',
            'intimacy_boundary',
            'ambiguous_intimate',
            'safety_blocked'
        )
    ),
    intimacy_boundary_confidence REAL NOT NULL DEFAULT 0.0 CHECK (
        intimacy_boundary_confidence BETWEEN 0.0 AND 1.0
    ),
    user_persona_id TEXT,
    platform_id TEXT,
    character_id TEXT,
    sensitivity TEXT NOT NULL DEFAULT 'unknown'
        CHECK (sensitivity IN ('unknown', 'public', 'private', 'secret')),
    themes_json TEXT NOT NULL DEFAULT '[]',
    platform_locked INTEGER NOT NULL DEFAULT 0 CHECK (platform_locked IN (0, 1)),
    platform_id_lock TEXT,
    scope_canonical TEXT
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
    created_at,
    intimacy_boundary,
    intimacy_boundary_confidence,
    user_persona_id,
    platform_id,
    character_id,
    sensitivity,
    themes_json,
    platform_locked,
    platform_id_lock,
    scope_canonical
)
SELECT
    id,
    user_id,
    conversation_id,
    workspace_id,
    source_message_start_seq,
    source_message_end_seq,
    CASE
        WHEN summary_kind = 'workspace_rollup' THEN 'character_rollup'
        ELSE summary_kind
    END,
    summary_text,
    source_object_ids_json,
    maya_score,
    model,
    hierarchy_level,
    created_at,
    intimacy_boundary,
    intimacy_boundary_confidence,
    user_persona_id,
    COALESCE(platform_id, 'default'),
    COALESCE(character_id, workspace_id),
    CASE
        WHEN summary_kind = 'workspace_rollup' AND sensitivity = 'unknown' THEN 'public'
        ELSE sensitivity
    END,
    themes_json,
    platform_locked,
    platform_id_lock,
    CASE
        WHEN summary_kind = 'workspace_rollup' THEN 'character'
        ELSE scope_canonical
    END
FROM summary_views;

DROP TABLE summary_views;
ALTER TABLE summary_views_new RENAME TO summary_views;

UPDATE memory_objects
SET payload_json = json_set(payload_json, '$.summary_kind', 'character_rollup'),
    character_id = COALESCE(character_id, workspace_id),
    platform_id = COALESCE(platform_id, 'default'),
    sensitivity = CASE
        WHEN sensitivity = 'unknown' AND privacy_level <= 1 THEN 'public'
        ELSE sensitivity
    END,
    scope_canonical = 'character'
WHERE object_type = 'summary_view'
  AND json_valid(payload_json)
  AND json_extract(payload_json, '$.summary_kind') = 'workspace_rollup';

CREATE INDEX IF NOT EXISTS idx_summary_views_user_kind
    ON summary_views(user_id, summary_kind, hierarchy_level, created_at);

CREATE INDEX IF NOT EXISTS idx_sv_user_kind_intimacy
    ON summary_views(user_id, summary_kind, intimacy_boundary, created_at);

CREATE INDEX IF NOT EXISTS idx_sv_character_rollup
    ON summary_views(user_id, user_persona_id, character_id, summary_kind, created_at DESC);
