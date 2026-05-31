CREATE TABLE IF NOT EXISTS memory_support_edges (
    _rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    memory_id TEXT NOT NULL REFERENCES memory_objects(id) ON DELETE CASCADE,
    support_kind TEXT NOT NULL DEFAULT 'direct' CHECK (
        support_kind IN ('direct', 'contextual_direct', 'inferred', 'weak_signal')
    ),
    evidence_polarity TEXT NOT NULL DEFAULT 'supports' CHECK (
        evidence_polarity IN ('supports', 'qualifies', 'contradicts')
    ),
    speaker_relation_to_subject TEXT NOT NULL DEFAULT 'unknown' CHECK (
        speaker_relation_to_subject IN (
            'self_report',
            'second_person_confirmation',
            'third_party_report',
            'assistant_inference',
            'behavioral_observation',
            'unknown'
        )
    ),
    confidence REAL NOT NULL DEFAULT 0.5 CHECK (confidence BETWEEN 0.0 AND 1.0),
    confidence_json TEXT NOT NULL DEFAULT '{}',
    rationale TEXT,
    status TEXT NOT NULL DEFAULT 'active' CHECK (
        status IN ('active', 'review_required', 'deleted')
    ),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_support_edges_user_memory
    ON memory_support_edges(user_id, memory_id, status, updated_at DESC, id ASC);

CREATE INDEX IF NOT EXISTS idx_memory_support_edges_user_kind
    ON memory_support_edges(user_id, support_kind, status, updated_at DESC, id ASC);

CREATE INDEX IF NOT EXISTS idx_memory_support_edges_user_polarity
    ON memory_support_edges(user_id, evidence_polarity, status, updated_at DESC, id ASC);

CREATE TRIGGER IF NOT EXISTS memory_support_edges_owner_bi
BEFORE INSERT ON memory_support_edges
BEGIN
    SELECT RAISE(ABORT, 'memory_support_edges user_id must match memory_objects.user_id')
    WHERE NOT EXISTS (
        SELECT 1
        FROM memory_objects
        WHERE memory_objects.id = new.memory_id
          AND memory_objects.user_id = new.user_id
    );
END;

CREATE TRIGGER IF NOT EXISTS memory_support_edges_owner_bu
BEFORE UPDATE OF user_id, memory_id ON memory_support_edges
BEGIN
    SELECT RAISE(ABORT, 'memory_support_edges user_id must match memory_objects.user_id')
    WHERE NOT EXISTS (
        SELECT 1
        FROM memory_objects
        WHERE memory_objects.id = new.memory_id
          AND memory_objects.user_id = new.user_id
    );
END;

CREATE TABLE IF NOT EXISTS memory_evidence_spans (
    _rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    support_edge_id TEXT NOT NULL REFERENCES memory_support_edges(id) ON DELETE CASCADE,
    memory_id TEXT NOT NULL REFERENCES memory_objects(id) ON DELETE CASCADE,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
    message_id TEXT REFERENCES messages(id) ON DELETE CASCADE,
    span_role TEXT NOT NULL CHECK (
        span_role IN ('source', 'trigger', 'qualifier', 'contradiction')
    ),
    quote_text TEXT NOT NULL CHECK (TRIM(quote_text) <> ''),
    char_start INTEGER,
    char_end INTEGER,
    seq INTEGER,
    occurred_at TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (char_start IS NULL OR char_start >= 0),
    CHECK (char_end IS NULL OR char_end >= 0),
    CHECK (char_start IS NULL OR char_end IS NULL OR char_start <= char_end)
);

CREATE INDEX IF NOT EXISTS idx_memory_evidence_spans_user_memory
    ON memory_evidence_spans(user_id, memory_id, span_role, created_at ASC, id ASC);

CREATE INDEX IF NOT EXISTS idx_memory_evidence_spans_user_edge
    ON memory_evidence_spans(user_id, support_edge_id, span_role, created_at ASC, id ASC);

CREATE INDEX IF NOT EXISTS idx_memory_evidence_spans_user_message
    ON memory_evidence_spans(user_id, message_id, span_role, created_at ASC, id ASC);

CREATE TRIGGER IF NOT EXISTS memory_evidence_spans_owner_bi
BEFORE INSERT ON memory_evidence_spans
BEGIN
    SELECT RAISE(ABORT, 'memory_evidence_spans support edge must match user_id and memory_id')
    WHERE NOT EXISTS (
        SELECT 1
        FROM memory_support_edges
        WHERE memory_support_edges.id = new.support_edge_id
          AND memory_support_edges.user_id = new.user_id
          AND memory_support_edges.memory_id = new.memory_id
    );

    SELECT RAISE(ABORT, 'memory_evidence_spans message must belong to user_id and conversation_id')
    WHERE new.message_id IS NOT NULL
      AND NOT EXISTS (
          SELECT 1
          FROM messages AS m
          JOIN conversations AS c ON c.id = m.conversation_id
          WHERE m.id = new.message_id
            AND c.user_id = new.user_id
            AND (new.conversation_id IS NULL OR new.conversation_id = m.conversation_id)
      );
END;

CREATE TRIGGER IF NOT EXISTS memory_evidence_spans_owner_bu
BEFORE UPDATE OF user_id, support_edge_id, memory_id, conversation_id, message_id ON memory_evidence_spans
BEGIN
    SELECT RAISE(ABORT, 'memory_evidence_spans support edge must match user_id and memory_id')
    WHERE NOT EXISTS (
        SELECT 1
        FROM memory_support_edges
        WHERE memory_support_edges.id = new.support_edge_id
          AND memory_support_edges.user_id = new.user_id
          AND memory_support_edges.memory_id = new.memory_id
    );

    SELECT RAISE(ABORT, 'memory_evidence_spans message must belong to user_id and conversation_id')
    WHERE new.message_id IS NOT NULL
      AND NOT EXISTS (
          SELECT 1
          FROM messages AS m
          JOIN conversations AS c ON c.id = m.conversation_id
          WHERE m.id = new.message_id
            AND c.user_id = new.user_id
            AND (new.conversation_id IS NULL OR new.conversation_id = m.conversation_id)
      );
END;
