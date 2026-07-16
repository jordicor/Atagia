ALTER TABLE worker_job_runs ADD COLUMN deferred_until TEXT;

CREATE INDEX IF NOT EXISTS idx_worker_job_runs_deferred_until
    ON worker_job_runs(status, deferred_until, queued_at ASC, job_id ASC);
