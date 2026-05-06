CREATE TABLE IF NOT EXISTS worker_control_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    mode TEXT NOT NULL DEFAULT 'active',
    reason TEXT,
    updated_at TEXT NOT NULL,
    updated_by TEXT,
    CHECK (mode IN ('active', 'pause_new_jobs', 'drain_and_pause', 'hard_pause'))
);
