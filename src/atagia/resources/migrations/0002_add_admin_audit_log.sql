CREATE TABLE IF NOT EXISTS admin_audit_log (
    _rowid INTEGER PRIMARY KEY,
    id TEXT NOT NULL UNIQUE,
    admin_user_id TEXT NOT NULL,
    action TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_admin_audit_admin_created
    ON admin_audit_log(admin_user_id, created_at);

CREATE INDEX IF NOT EXISTS idx_admin_audit_target
    ON admin_audit_log(target_type, target_id, created_at);
