ALTER TABLE worker_job_runs ADD COLUMN transient_defer_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE worker_job_runs ADD COLUMN first_deferred_at TEXT;
ALTER TABLE worker_job_runs ADD COLUMN last_deferred_at TEXT;

CREATE INDEX IF NOT EXISTS idx_worker_job_runs_transient_defer_budget
    ON worker_job_runs(status, transient_defer_count, first_deferred_at, queued_at ASC, job_id ASC);
