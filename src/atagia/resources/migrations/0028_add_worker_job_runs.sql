CREATE TABLE IF NOT EXISTS worker_job_runs (
    _rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL UNIQUE,
    stream_name TEXT NOT NULL,
    job_type TEXT NOT NULL,
    user_id TEXT NOT NULL,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
    source_message_ids_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    source_token_estimate INTEGER,
    size_bucket TEXT,
    queued_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    last_heartbeat_at TEXT,
    duration_ms REAL,
    error_class TEXT,
    error_message TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    CHECK (status IN (
        'queued',
        'running',
        'retrying',
        'succeeded',
        'skipped',
        'failed',
        'dead_lettered',
        'cancelled'
    )),
    CHECK (attempt_count >= 0),
    CHECK (duration_ms IS NULL OR duration_ms >= 0),
    CHECK (source_token_estimate IS NULL OR source_token_estimate >= 0)
);

CREATE INDEX IF NOT EXISTS idx_worker_job_runs_user_status
    ON worker_job_runs(user_id, status, queued_at DESC, job_id ASC);

CREATE INDEX IF NOT EXISTS idx_worker_job_runs_conversation_status
    ON worker_job_runs(user_id, conversation_id, status, queued_at DESC, job_id ASC);

CREATE INDEX IF NOT EXISTS idx_worker_job_runs_type_finished
    ON worker_job_runs(job_type, size_bucket, finished_at DESC, job_id ASC);

CREATE INDEX IF NOT EXISTS idx_worker_job_runs_user_conversation_queued
    ON worker_job_runs(user_id, conversation_id, queued_at DESC, job_id ASC);
