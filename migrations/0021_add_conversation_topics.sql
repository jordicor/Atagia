CREATE TABLE IF NOT EXISTS conversation_topics (
    _rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    parent_topic_id TEXT REFERENCES conversation_topics(id) ON DELETE SET NULL,
    status TEXT NOT NULL DEFAULT 'active',
    title TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    active_goal TEXT,
    open_questions_json TEXT NOT NULL DEFAULT '[]',
    decisions_json TEXT NOT NULL DEFAULT '[]',
    artifact_ids_json TEXT NOT NULL DEFAULT '[]',
    source_message_start_seq INTEGER,
    source_message_end_seq INTEGER,
    last_touched_seq INTEGER,
    last_touched_at TEXT,
    confidence REAL NOT NULL DEFAULT 0.5,
    privacy_level INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (status IN ('active', 'parked', 'closed')),
    CHECK (confidence BETWEEN 0.0 AND 1.0),
    CHECK (privacy_level BETWEEN 0 AND 3),
    CHECK (source_message_start_seq IS NULL OR source_message_start_seq >= 0),
    CHECK (source_message_end_seq IS NULL OR source_message_end_seq >= 0),
    CHECK (last_touched_seq IS NULL OR last_touched_seq >= 0)
);

CREATE INDEX IF NOT EXISTS conversation_topics_user_conversation_status_idx
    ON conversation_topics(user_id, conversation_id, status, last_touched_at DESC, id ASC);

CREATE INDEX IF NOT EXISTS conversation_topics_user_parent_idx
    ON conversation_topics(user_id, parent_topic_id, id ASC);

CREATE TABLE IF NOT EXISTS conversation_topic_events (
    _rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    topic_id TEXT REFERENCES conversation_topics(id) ON DELETE SET NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    source_message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL,
    CHECK (event_type IN ('created', 'updated', 'parked', 'reopened', 'closed', 'source_linked'))
);

CREATE INDEX IF NOT EXISTS conversation_topic_events_user_conversation_idx
    ON conversation_topic_events(user_id, conversation_id, created_at ASC, id ASC);

CREATE INDEX IF NOT EXISTS conversation_topic_events_user_topic_idx
    ON conversation_topic_events(user_id, topic_id, created_at ASC, id ASC);

CREATE TABLE IF NOT EXISTS conversation_topic_sources (
    _rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    topic_id TEXT NOT NULL REFERENCES conversation_topics(id) ON DELETE CASCADE,
    source_kind TEXT NOT NULL,
    source_id TEXT NOT NULL,
    relation_kind TEXT NOT NULL DEFAULT 'evidence',
    created_at TEXT NOT NULL,
    CHECK (source_kind IN ('message', 'artifact', 'memory_object')),
    CHECK (relation_kind IN ('evidence', 'artifact', 'decision', 'open_question', 'constraint'))
);

CREATE UNIQUE INDEX IF NOT EXISTS conversation_topic_sources_unique_idx
    ON conversation_topic_sources(user_id, topic_id, source_kind, source_id, relation_kind);

CREATE INDEX IF NOT EXISTS conversation_topic_sources_user_topic_idx
    ON conversation_topic_sources(user_id, topic_id, created_at ASC, id ASC);
