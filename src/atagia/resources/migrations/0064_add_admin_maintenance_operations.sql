-- Durable coordination between long-running admin mutations and authoritative
-- selected-transcript replacement.

CREATE TABLE admin_maintenance_operations (
    id TEXT PRIMARY KEY,
    operation_kind TEXT NOT NULL,
    recovery_key TEXT NOT NULL,
    scope_kind TEXT NOT NULL CHECK (scope_kind IN ('user', 'global')),
    user_id TEXT REFERENCES users(id) ON DELETE CASCADE,
    lifecycle_epoch TEXT,
    derivation_revision INTEGER CHECK (
        derivation_revision IS NULL OR derivation_revision >= 0
    ),
    owner_token TEXT NOT NULL,
    heartbeat_at TEXT NOT NULL,
    lease_expires_at TEXT NOT NULL,
    phase TEXT NOT NULL DEFAULT 'prepared' CHECK (phase IN ('prepared', 'dirty')),
    status TEXT NOT NULL CHECK (
        status IN ('active', 'succeeded', 'failed', 'remediation_required')
    ),
    error_class TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    CHECK (
        (scope_kind = 'user'
         AND user_id IS NOT NULL
         AND lifecycle_epoch IS NOT NULL
         AND derivation_revision IS NOT NULL)
        OR
        (scope_kind = 'global'
         AND user_id IS NULL
         AND lifecycle_epoch IS NULL
         AND derivation_revision IS NULL)
    )
);

CREATE UNIQUE INDEX idx_admin_maintenance_active_user
    ON admin_maintenance_operations(user_id)
    WHERE status IN ('active', 'remediation_required') AND scope_kind = 'user';

CREATE UNIQUE INDEX idx_admin_maintenance_active_global
    ON admin_maintenance_operations(scope_kind)
    WHERE status IN ('active', 'remediation_required') AND scope_kind = 'global';

CREATE INDEX idx_admin_maintenance_status_updated
    ON admin_maintenance_operations(status, updated_at, id);

CREATE TABLE admin_maintenance_effects (
    operation_id TEXT NOT NULL
        REFERENCES admin_maintenance_operations(id) ON DELETE CASCADE,
    effect_kind TEXT NOT NULL CHECK (effect_kind IN ('delete_embedding')),
    target_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'completed')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    PRIMARY KEY (operation_id, effect_kind, target_id)
);

CREATE INDEX idx_admin_maintenance_effects_pending
    ON admin_maintenance_effects(operation_id, status, effect_kind, target_id);

ALTER TABLE worker_job_runs
    ADD COLUMN maintenance_operation_id TEXT
    REFERENCES admin_maintenance_operations(id) ON DELETE SET NULL;

CREATE INDEX idx_worker_job_runs_maintenance_operation
    ON worker_job_runs(maintenance_operation_id, status, job_type, queued_at, job_id);

-- These triggers are the last-line, cross-process invariant. Application-side
-- checks provide stable errors, while the triggers make bypassing the
-- coordination protocol impossible for either writer.
CREATE TRIGGER admin_maintenance_block_selected_transcript_insert
BEFORE INSERT ON conversation_transcript_selections
WHEN new.state IN ('preparing', 'rebuilding', 'remediation_required')
 AND EXISTS (
    SELECT 1
    FROM admin_maintenance_operations AS operation
    WHERE (
        operation.status = 'remediation_required'
        OR (
            operation.status = 'active'
            AND (
                operation.phase = 'dirty'
                OR julianday(operation.lease_expires_at) > julianday('now')
            )
        )
    )
      AND (operation.scope_kind = 'global' OR operation.user_id = new.user_id)
 )
BEGIN
    SELECT RAISE(ABORT, 'admin maintenance operation blocks selected transcript replacement');
END;

CREATE TRIGGER admin_maintenance_block_selected_transcript_update
BEFORE UPDATE ON conversation_transcript_selections
WHEN new.state IN ('preparing', 'rebuilding', 'remediation_required')
 AND EXISTS (
    SELECT 1
    FROM admin_maintenance_operations AS operation
    WHERE (
        operation.status = 'remediation_required'
        OR (
            operation.status = 'active'
            AND (
                operation.phase = 'dirty'
                OR julianday(operation.lease_expires_at) > julianday('now')
            )
        )
    )
      AND (operation.scope_kind = 'global' OR operation.user_id = new.user_id)
 )
BEGIN
    SELECT RAISE(ABORT, 'admin maintenance operation blocks selected transcript replacement');
END;

CREATE TRIGGER selected_transcript_block_admin_maintenance_insert
BEFORE INSERT ON admin_maintenance_operations
WHEN new.status = 'active'
 AND EXISTS (
    SELECT 1
    FROM conversation_transcript_selections AS selection
    WHERE selection.state IN ('rebuilding', 'remediation_required')
      AND (new.scope_kind = 'global' OR selection.user_id = new.user_id)
 )
BEGIN
    SELECT RAISE(ABORT, 'selected transcript rebuild blocks admin maintenance operation');
END;

CREATE TRIGGER selected_transcript_block_admin_maintenance_reactivate
BEFORE UPDATE OF status ON admin_maintenance_operations
WHEN new.status = 'active'
 AND old.status != 'active'
 AND EXISTS (
    SELECT 1
    FROM conversation_transcript_selections AS selection
    WHERE selection.state IN ('rebuilding', 'remediation_required')
      AND (new.scope_kind = 'global' OR selection.user_id = new.user_id)
 )
BEGIN
    SELECT RAISE(ABORT, 'selected transcript rebuild blocks admin maintenance operation');
END;
