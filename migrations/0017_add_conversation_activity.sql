CREATE TABLE IF NOT EXISTS conversation_activity_stats (
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    workspace_id TEXT REFERENCES workspaces(id) ON DELETE SET NULL,
    assistant_mode_id TEXT NOT NULL REFERENCES assistant_modes(id) ON DELETE RESTRICT,
    timezone TEXT NOT NULL DEFAULT 'UTC',
    first_message_at TEXT,
    last_message_at TEXT,
    last_user_message_at TEXT,
    message_count INTEGER NOT NULL DEFAULT 0,
    user_message_count INTEGER NOT NULL DEFAULT 0,
    assistant_message_count INTEGER NOT NULL DEFAULT 0,
    retrieval_count INTEGER NOT NULL DEFAULT 0,
    active_day_count INTEGER NOT NULL DEFAULT 0,
    recent_1d_message_count INTEGER NOT NULL DEFAULT 0,
    recent_7d_message_count INTEGER NOT NULL DEFAULT 0,
    recent_30d_message_count INTEGER NOT NULL DEFAULT 0,
    weekday_histogram_json TEXT NOT NULL DEFAULT '[]',
    hour_histogram_json TEXT NOT NULL DEFAULT '[]',
    hour_of_week_histogram_json TEXT NOT NULL DEFAULT '[]',
    return_interval_histogram_json TEXT NOT NULL DEFAULT '[]',
    avg_return_interval_minutes REAL,
    median_return_interval_minutes REAL,
    p90_return_interval_minutes REAL,
    main_thread_score REAL NOT NULL DEFAULT 0.0,
    likely_soon_score REAL NOT NULL DEFAULT 0.0,
    return_habit_confidence REAL NOT NULL DEFAULT 0.0,
    schedule_pattern_kind TEXT NOT NULL DEFAULT 'inactive',
    activity_version INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, conversation_id)
);

CREATE INDEX IF NOT EXISTS idx_conversation_activity_user_hot
    ON conversation_activity_stats(user_id, likely_soon_score DESC, main_thread_score DESC, last_message_at DESC, conversation_id ASC);

CREATE INDEX IF NOT EXISTS idx_conversation_activity_user_workspace
    ON conversation_activity_stats(user_id, workspace_id, likely_soon_score DESC, last_message_at DESC, conversation_id ASC);

CREATE INDEX IF NOT EXISTS idx_conversation_activity_user_mode
    ON conversation_activity_stats(user_id, assistant_mode_id, likely_soon_score DESC, last_message_at DESC, conversation_id ASC);
