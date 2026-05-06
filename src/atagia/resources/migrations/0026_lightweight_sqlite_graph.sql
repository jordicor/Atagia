CREATE TABLE IF NOT EXISTS graph_projection_runs (
    _rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
    source_message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
    source_memory_ids_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'running',
    entity_count INTEGER NOT NULL DEFAULT 0,
    mention_count INTEGER NOT NULL DEFAULT 0,
    relationship_count INTEGER NOT NULL DEFAULT 0,
    skipped_count INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    finished_at TEXT,
    CHECK (status IN ('running', 'completed', 'failed')),
    CHECK (entity_count >= 0),
    CHECK (mention_count >= 0),
    CHECK (relationship_count >= 0),
    CHECK (skipped_count >= 0)
);

CREATE INDEX IF NOT EXISTS graph_projection_runs_user_conversation_idx
    ON graph_projection_runs(user_id, conversation_id, created_at DESC, id ASC);

CREATE INDEX IF NOT EXISTS graph_projection_runs_user_status_idx
    ON graph_projection_runs(user_id, status, created_at DESC, id ASC);

CREATE TABLE IF NOT EXISTS graph_entities (
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

CREATE INDEX IF NOT EXISTS graph_entities_user_type_status_idx
    ON graph_entities(user_id, entity_type, status, updated_at DESC, id ASC);

CREATE INDEX IF NOT EXISTS graph_entities_user_workspace_idx
    ON graph_entities(user_id, workspace_id, status, id ASC);

CREATE INDEX IF NOT EXISTS graph_entities_user_conversation_idx
    ON graph_entities(user_id, conversation_id, status, id ASC);

CREATE TABLE IF NOT EXISTS graph_entity_mentions (
    _rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    entity_id TEXT REFERENCES graph_entities(id) ON DELETE CASCADE,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
    message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
    memory_id TEXT REFERENCES memory_objects(id) ON DELETE CASCADE,
    projection_run_id TEXT REFERENCES graph_projection_runs(id) ON DELETE SET NULL,
    source_kind TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_signature TEXT NOT NULL,
    surface_text TEXT NOT NULL,
    surface_key TEXT NOT NULL,
    span_start INTEGER,
    span_end INTEGER,
    quote_hash TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 0.5,
    status TEXT NOT NULL DEFAULT 'active',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (source_kind IN ('message', 'artifact', 'artifact_chunk', 'memory_object')),
    CHECK (source_id <> ''),
    CHECK (source_signature <> ''),
    CHECK (surface_text <> ''),
    CHECK (surface_key <> ''),
    CHECK (span_start IS NULL OR span_start >= 0),
    CHECK (span_end IS NULL OR span_end >= 0),
    CHECK (span_start IS NULL OR span_end IS NULL OR span_start <= span_end),
    CHECK (confidence BETWEEN 0.0 AND 1.0),
    CHECK (status IN ('active', 'review_required', 'archived', 'deleted'))
);

CREATE UNIQUE INDEX IF NOT EXISTS graph_entity_mentions_source_signature_idx
    ON graph_entity_mentions(user_id, source_signature);

CREATE INDEX IF NOT EXISTS graph_entity_mentions_user_entity_idx
    ON graph_entity_mentions(user_id, entity_id, created_at ASC, id ASC);

CREATE INDEX IF NOT EXISTS graph_entity_mentions_user_source_idx
    ON graph_entity_mentions(user_id, source_kind, source_id, created_at ASC, id ASC);

CREATE INDEX IF NOT EXISTS graph_entity_mentions_user_conversation_idx
    ON graph_entity_mentions(user_id, conversation_id, created_at ASC, id ASC);

CREATE TABLE IF NOT EXISTS graph_entity_aliases (
    _rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    entity_id TEXT NOT NULL REFERENCES graph_entities(id) ON DELETE CASCADE,
    source_mention_id TEXT REFERENCES graph_entity_mentions(id) ON DELETE SET NULL,
    surface_text TEXT NOT NULL,
    surface_key TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    status TEXT NOT NULL DEFAULT 'active',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (surface_text <> ''),
    CHECK (surface_key <> ''),
    CHECK (confidence BETWEEN 0.0 AND 1.0),
    CHECK (status IN ('active', 'review_required', 'archived', 'deleted'))
);

CREATE UNIQUE INDEX IF NOT EXISTS graph_entity_aliases_unique_idx
    ON graph_entity_aliases(user_id, entity_id, surface_key);

CREATE INDEX IF NOT EXISTS graph_entity_aliases_user_surface_idx
    ON graph_entity_aliases(user_id, surface_key, status, entity_id);

CREATE TABLE IF NOT EXISTS graph_relationships (
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

CREATE UNIQUE INDEX IF NOT EXISTS graph_relationships_user_dedupe_idx
    ON graph_relationships(user_id, dedupe_key);

CREATE INDEX IF NOT EXISTS graph_relationships_user_source_idx
    ON graph_relationships(user_id, source_entity_id, status, updated_at DESC, id ASC);

CREATE INDEX IF NOT EXISTS graph_relationships_user_target_idx
    ON graph_relationships(user_id, target_entity_id, status, updated_at DESC, id ASC);

CREATE INDEX IF NOT EXISTS graph_relationships_user_predicate_idx
    ON graph_relationships(user_id, predicate, status, updated_at DESC, id ASC);

CREATE TABLE IF NOT EXISTS graph_relationship_sources (
    _rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    relationship_id TEXT NOT NULL REFERENCES graph_relationships(id) ON DELETE CASCADE,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
    message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
    memory_id TEXT REFERENCES memory_objects(id) ON DELETE CASCADE,
    projection_run_id TEXT REFERENCES graph_projection_runs(id) ON DELETE SET NULL,
    source_kind TEXT NOT NULL,
    source_id TEXT NOT NULL,
    quote_hash TEXT NOT NULL DEFAULT '',
    evidence_quote TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    CHECK (source_kind IN ('message', 'artifact', 'artifact_chunk', 'memory_object')),
    CHECK (source_id <> '')
);

CREATE UNIQUE INDEX IF NOT EXISTS graph_relationship_sources_unique_idx
    ON graph_relationship_sources(user_id, relationship_id, source_kind, source_id, quote_hash);

CREATE INDEX IF NOT EXISTS graph_relationship_sources_user_relationship_idx
    ON graph_relationship_sources(user_id, relationship_id, created_at ASC, id ASC);

CREATE INDEX IF NOT EXISTS graph_relationship_sources_user_source_idx
    ON graph_relationship_sources(user_id, source_kind, source_id, created_at ASC, id ASC);

CREATE INDEX IF NOT EXISTS graph_relationship_sources_user_conversation_idx
    ON graph_relationship_sources(user_id, conversation_id, created_at ASC, id ASC);
