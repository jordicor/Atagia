-- atagia:foreign_keys_off

CREATE TABLE graph_entities_new (
    _rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    workspace_id TEXT REFERENCES workspaces(id) ON DELETE CASCADE,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
    assistant_mode_id TEXT REFERENCES assistant_modes(id) ON DELETE CASCADE,
    entity_type TEXT NOT NULL,
    display_name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 0.5,
    status TEXT NOT NULL DEFAULT 'active',
    privacy_level INTEGER NOT NULL DEFAULT 0,
    intimacy_boundary TEXT NOT NULL DEFAULT 'ordinary',
    intimacy_boundary_confidence REAL NOT NULL DEFAULT 0.0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (entity_type <> ''),
    CHECK (display_name <> ''),
    CHECK (confidence BETWEEN 0.0 AND 1.0),
    CHECK (status IN ('active', 'review_required', 'merged', 'archived', 'deleted')),
    CHECK (privacy_level BETWEEN 0 AND 3),
    CHECK (
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
    CHECK (intimacy_boundary_confidence BETWEEN 0.0 AND 1.0)
);

INSERT INTO graph_entities_new(
    _rowid,
    id,
    user_id,
    workspace_id,
    conversation_id,
    assistant_mode_id,
    entity_type,
    display_name,
    description,
    confidence,
    status,
    privacy_level,
    intimacy_boundary,
    intimacy_boundary_confidence,
    metadata_json,
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
    entity_type,
    display_name,
    description,
    confidence,
    status,
    privacy_level,
    intimacy_boundary,
    intimacy_boundary_confidence,
    metadata_json,
    created_at,
    updated_at
FROM graph_entities;

CREATE TABLE graph_relationships_new (
    _rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    source_entity_id TEXT NOT NULL REFERENCES graph_entities(id) ON DELETE CASCADE,
    target_entity_id TEXT REFERENCES graph_entities(id) ON DELETE CASCADE,
    target_value_json TEXT,
    predicate TEXT NOT NULL,
    direction TEXT NOT NULL DEFAULT 'directed',
    scope TEXT NOT NULL DEFAULT 'conversation',
    workspace_id TEXT REFERENCES workspaces(id) ON DELETE CASCADE,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
    assistant_mode_id TEXT REFERENCES assistant_modes(id) ON DELETE CASCADE,
    confidence REAL NOT NULL DEFAULT 0.5,
    status TEXT NOT NULL DEFAULT 'active',
    valid_from TEXT,
    valid_to TEXT,
    privacy_level INTEGER NOT NULL DEFAULT 0,
    intimacy_boundary TEXT NOT NULL DEFAULT 'ordinary',
    intimacy_boundary_confidence REAL NOT NULL DEFAULT 0.0,
    supersedes_relationship_id TEXT REFERENCES graph_relationships(id) ON DELETE SET NULL,
    dedupe_key TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (target_entity_id IS NOT NULL OR target_value_json IS NOT NULL),
    CHECK (predicate <> ''),
    CHECK (direction IN ('directed', 'symmetric')),
    CHECK (
        scope IN (
            'global_user',
            'assistant_mode',
            'workspace',
            'conversation',
            'ephemeral_session'
        )
    ),
    CHECK (confidence BETWEEN 0.0 AND 1.0),
    CHECK (status IN ('active', 'review_required', 'superseded', 'conflicted', 'archived', 'deleted')),
    CHECK (privacy_level BETWEEN 0 AND 3),
    CHECK (
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
    CHECK (intimacy_boundary_confidence BETWEEN 0.0 AND 1.0)
);

INSERT INTO graph_relationships_new(
    _rowid,
    id,
    user_id,
    source_entity_id,
    target_entity_id,
    target_value_json,
    predicate,
    direction,
    scope,
    workspace_id,
    conversation_id,
    assistant_mode_id,
    confidence,
    status,
    valid_from,
    valid_to,
    privacy_level,
    intimacy_boundary,
    intimacy_boundary_confidence,
    supersedes_relationship_id,
    dedupe_key,
    metadata_json,
    created_at,
    updated_at
)
SELECT
    _rowid,
    id,
    user_id,
    source_entity_id,
    target_entity_id,
    target_value_json,
    predicate,
    direction,
    scope,
    workspace_id,
    conversation_id,
    assistant_mode_id,
    confidence,
    status,
    valid_from,
    valid_to,
    privacy_level,
    intimacy_boundary,
    intimacy_boundary_confidence,
    supersedes_relationship_id,
    dedupe_key,
    metadata_json,
    created_at,
    updated_at
FROM graph_relationships;

DROP TABLE graph_relationships;
ALTER TABLE graph_relationships_new RENAME TO graph_relationships;

DROP TABLE graph_entities;
ALTER TABLE graph_entities_new RENAME TO graph_entities;

CREATE INDEX IF NOT EXISTS graph_entities_user_type_status_idx
    ON graph_entities(user_id, entity_type, status, updated_at DESC, id ASC);

CREATE INDEX IF NOT EXISTS graph_entities_user_workspace_idx
    ON graph_entities(user_id, workspace_id, status, id ASC);

CREATE INDEX IF NOT EXISTS graph_entities_user_conversation_idx
    ON graph_entities(user_id, conversation_id, status, id ASC);

CREATE UNIQUE INDEX IF NOT EXISTS graph_relationships_user_dedupe_idx
    ON graph_relationships(user_id, dedupe_key);

CREATE INDEX IF NOT EXISTS graph_relationships_user_source_idx
    ON graph_relationships(user_id, source_entity_id, status, updated_at DESC, id ASC);

CREATE INDEX IF NOT EXISTS graph_relationships_user_target_idx
    ON graph_relationships(user_id, target_entity_id, status, updated_at DESC, id ASC);

CREATE INDEX IF NOT EXISTS graph_relationships_user_predicate_idx
    ON graph_relationships(user_id, predicate, status, updated_at DESC, id ASC);
