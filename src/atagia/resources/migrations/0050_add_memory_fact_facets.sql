CREATE TABLE IF NOT EXISTS memory_fact_facets (
    _rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
    memory_id TEXT NOT NULL REFERENCES memory_objects(id) ON DELETE CASCADE,
    source_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    source_span_id TEXT NOT NULL REFERENCES memory_evidence_spans(id) ON DELETE CASCADE,
    source_hash TEXT NOT NULL CHECK (TRIM(source_hash) <> ''),
    subject_surface TEXT NOT NULL CHECK (TRIM(subject_surface) <> ''),
    subject_cluster_id TEXT,
    facet_label TEXT NOT NULL CHECK (TRIM(facet_label) <> ''),
    value_text TEXT NOT NULL CHECK (TRIM(value_text) <> ''),
    value_norm_key TEXT NOT NULL CHECK (TRIM(value_norm_key) <> ''),
    value_type TEXT NOT NULL DEFAULT 'text' CHECK (TRIM(value_type) <> ''),
    assertion_kind TEXT NOT NULL CHECK (TRIM(assertion_kind) <> ''),
    list_group_key TEXT,
    support_kind TEXT NOT NULL CHECK (
        support_kind IN ('direct', 'contextual_direct', 'inferred', 'weak_signal')
    ),
    observed_at TEXT NOT NULL,
    valid_from TEXT,
    valid_to TEXT,
    current_state INTEGER NOT NULL DEFAULT 1 CHECK (current_state IN (0, 1)),
    supersedes_fact_id TEXT REFERENCES memory_fact_facets(id) ON DELETE SET NULL,
    temporal_phrase TEXT,
    temporal_anchor_at TEXT,
    resolved_interval_start TEXT,
    resolved_interval_end TEXT,
    temporal_granularity TEXT,
    temporal_resolution_type TEXT,
    temporal_confidence REAL CHECK (
        temporal_confidence IS NULL
        OR temporal_confidence BETWEEN 0.0 AND 1.0
    ),
    language_code TEXT,
    confidence REAL NOT NULL DEFAULT 0.5 CHECK (confidence BETWEEN 0.0 AND 1.0),
    schema_version INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_memory_fact_facets_source_value
    ON memory_fact_facets(
        user_id,
        memory_id,
        source_span_id,
        facet_label,
        value_norm_key,
        assertion_kind
    );

CREATE INDEX IF NOT EXISTS idx_memory_fact_facets_user_conversation
    ON memory_fact_facets(user_id, conversation_id, created_at DESC, id ASC);

CREATE INDEX IF NOT EXISTS idx_memory_fact_facets_user_memory
    ON memory_fact_facets(user_id, memory_id, created_at DESC, id ASC);

CREATE INDEX IF NOT EXISTS idx_memory_fact_facets_user_facet_value
    ON memory_fact_facets(user_id, facet_label, value_norm_key, created_at DESC, id ASC);

CREATE INDEX IF NOT EXISTS idx_memory_fact_facets_user_source_span
    ON memory_fact_facets(user_id, source_span_id, created_at DESC, id ASC);

CREATE INDEX IF NOT EXISTS idx_memory_fact_facets_user_current
    ON memory_fact_facets(user_id, current_state, facet_label, created_at DESC, id ASC);

CREATE TRIGGER IF NOT EXISTS memory_fact_facets_owner_bi
BEFORE INSERT ON memory_fact_facets
BEGIN
    SELECT RAISE(ABORT, 'memory_fact_facets user_id must match memory_objects.user_id')
    WHERE NOT EXISTS (
        SELECT 1
        FROM memory_objects
        WHERE memory_objects.id = new.memory_id
          AND memory_objects.user_id = new.user_id
    );

    SELECT RAISE(ABORT, 'memory_fact_facets source span must match user_id, memory_id, and source_message_id')
    WHERE NOT EXISTS (
        SELECT 1
        FROM memory_evidence_spans
        WHERE memory_evidence_spans.id = new.source_span_id
          AND memory_evidence_spans.user_id = new.user_id
          AND memory_evidence_spans.memory_id = new.memory_id
          AND memory_evidence_spans.message_id = new.source_message_id
          AND memory_evidence_spans.span_role = 'source'
    );

    SELECT RAISE(ABORT, 'memory_fact_facets conversation must match source message owner')
    WHERE new.conversation_id IS NOT NULL
      AND NOT EXISTS (
          SELECT 1
          FROM messages AS m
          JOIN conversations AS c ON c.id = m.conversation_id
          WHERE m.id = new.source_message_id
            AND c.user_id = new.user_id
            AND c.id = new.conversation_id
      );
END;

CREATE TRIGGER IF NOT EXISTS memory_fact_facets_owner_bu
BEFORE UPDATE OF user_id, conversation_id, memory_id, source_message_id, source_span_id ON memory_fact_facets
BEGIN
    SELECT RAISE(ABORT, 'memory_fact_facets user_id must match memory_objects.user_id')
    WHERE NOT EXISTS (
        SELECT 1
        FROM memory_objects
        WHERE memory_objects.id = new.memory_id
          AND memory_objects.user_id = new.user_id
    );

    SELECT RAISE(ABORT, 'memory_fact_facets source span must match user_id, memory_id, and source_message_id')
    WHERE NOT EXISTS (
        SELECT 1
        FROM memory_evidence_spans
        WHERE memory_evidence_spans.id = new.source_span_id
          AND memory_evidence_spans.user_id = new.user_id
          AND memory_evidence_spans.memory_id = new.memory_id
          AND memory_evidence_spans.message_id = new.source_message_id
          AND memory_evidence_spans.span_role = 'source'
    );

    SELECT RAISE(ABORT, 'memory_fact_facets conversation must match source message owner')
    WHERE new.conversation_id IS NOT NULL
      AND NOT EXISTS (
          SELECT 1
          FROM messages AS m
          JOIN conversations AS c ON c.id = m.conversation_id
          WHERE m.id = new.source_message_id
            AND c.user_id = new.user_id
            AND c.id = new.conversation_id
      );
END;
