CREATE TABLE IF NOT EXISTS evaluation_metrics (
    id TEXT PRIMARY KEY,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    sample_count INTEGER NOT NULL DEFAULT 0,
    user_id TEXT,
    assistant_mode_id TEXT,
    workspace_id TEXT,
    time_bucket TEXT NOT NULL,
    computed_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_eval_metrics_name_bucket
    ON evaluation_metrics(metric_name, time_bucket);

CREATE INDEX IF NOT EXISTS idx_eval_metrics_user
    ON evaluation_metrics(user_id, metric_name);

CREATE UNIQUE INDEX IF NOT EXISTS idx_eval_metrics_unique
    ON evaluation_metrics(
        metric_name,
        time_bucket,
        COALESCE(user_id, ''),
        COALESCE(assistant_mode_id, ''),
        COALESCE(workspace_id, '')
    );
