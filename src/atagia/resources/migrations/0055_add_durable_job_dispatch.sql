-- atagia:foreign_keys_off

-- Durable per-user lifecycle identity. This row intentionally survives deletion
-- of the canonical user until erasure cleanup and retention have completed.
CREATE TABLE user_lifecycles (
    user_id TEXT PRIMARY KEY,
    lifecycle_epoch TEXT NOT NULL UNIQUE,
    lifecycle_cleanup_key TEXT NOT NULL UNIQUE,
    cache_revision INTEGER NOT NULL DEFAULT 0 CHECK (cache_revision >= 0),
    state TEXT NOT NULL DEFAULT 'active' CHECK (
        state IN ('active', 'erasing', 'cleanup_pending', 'erased')
    ),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    revoked_at TEXT,
    cleanup_completed_at TEXT,
    last_cleanup_error TEXT
);

INSERT INTO user_lifecycles(
    user_id,
    lifecycle_epoch,
    lifecycle_cleanup_key,
    cache_revision,
    state,
    created_at,
    updated_at
)
SELECT
    id,
    'ule_' || lower(hex(randomblob(16))),
    'ulk_' || lower(hex(randomblob(24))),
    0,
    CASE WHEN deleted_at IS NULL THEN 'active' ELSE 'erased' END,
    created_at,
    updated_at
FROM users;

-- Global administrative jobs still require an immutable lifecycle/fence owner,
-- but this internal principal must not appear in the canonical users table.
INSERT INTO user_lifecycles(
    user_id,
    lifecycle_epoch,
    lifecycle_cleanup_key,
    cache_revision,
    state,
    created_at,
    updated_at
) VALUES (
    'atagia_system',
    'ule_' || lower(hex(randomblob(16))),
    'ulk_' || lower(hex(randomblob(24))),
    0,
    'active',
    strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now'),
    strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
);

CREATE INDEX idx_user_lifecycles_state
    ON user_lifecycles(state, updated_at, user_id);

-- No legacy nonterminal row has a complete immutable envelope. Refuse to guess
-- one at cutover; operators must drain or explicitly cancel such work first.
CREATE TEMP TABLE worker_job_recovery_cutover_guard(nonterminal_count INTEGER NOT NULL);
CREATE TEMP TRIGGER worker_job_recovery_cutover_must_be_drained
BEFORE INSERT ON worker_job_recovery_cutover_guard
WHEN NEW.nonterminal_count > 0
BEGIN
    SELECT RAISE(
        ABORT,
        'migration 0055 requires all legacy worker_job_runs to be terminal; drain or cancel queued/running/retrying jobs before upgrading'
    );
END;
INSERT INTO worker_job_recovery_cutover_guard(nonterminal_count)
SELECT COUNT(*)
FROM worker_job_runs
WHERE status IN ('queued', 'running', 'retrying');
DROP TRIGGER worker_job_recovery_cutover_must_be_drained;
DROP TABLE worker_job_recovery_cutover_guard;

PRAGMA legacy_alter_table = ON;
ALTER TABLE worker_job_runs RENAME TO worker_job_runs_legacy;

CREATE TABLE worker_job_runs (
    _rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL UNIQUE,
    stream_name TEXT NOT NULL,
    target_backend TEXT NOT NULL,
    job_type TEXT NOT NULL,
    user_id TEXT NOT NULL,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
    parent_job_id TEXT,
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
    user_persona_id TEXT,
    platform_id TEXT,
    character_id TEXT,
    incognito_snapshot INTEGER NOT NULL DEFAULT 0 CHECK (incognito_snapshot IN (0, 1)),
    remember_across_chats_snapshot INTEGER NOT NULL DEFAULT 1 CHECK (remember_across_chats_snapshot IN (0, 1)),
    remember_across_devices_snapshot INTEGER NOT NULL DEFAULT 1 CHECK (remember_across_devices_snapshot IN (0, 1)),
    temporary_snapshot INTEGER NOT NULL DEFAULT 0 CHECK (temporary_snapshot IN (0, 1)),
    purge_on_close_snapshot INTEGER NOT NULL DEFAULT 0 CHECK (purge_on_close_snapshot IN (0, 1)),
    policy_snapshot_json TEXT NOT NULL DEFAULT '{}',
    deferred_until TEXT,
    transient_defer_count INTEGER NOT NULL DEFAULT 0,
    first_deferred_at TEXT,
    last_deferred_at TEXT,
    envelope_schema_version INTEGER,
    recovery_envelope_json TEXT,
    lifecycle_epoch TEXT NOT NULL,
    lifecycle_cleanup_key TEXT NOT NULL,
    dispatch_token TEXT,
    dispatch_visibility_deadline TEXT,
    dispatch_attempt_count INTEGER NOT NULL DEFAULT 0,
    execution_owner TEXT,
    execution_fence INTEGER NOT NULL DEFAULT 0,
    execution_lease_expires_at TEXT,
    terminal_diagnostics_json TEXT NOT NULL DEFAULT '{}',
    CHECK (status IN (
        'queued',
        'awaiting_claim',
        'running',
        'retrying',
        'deferred',
        'succeeded',
        'skipped',
        'failed',
        'dead_lettered',
        'cancelled'
    )),
    CHECK (attempt_count >= 0),
    CHECK (dispatch_attempt_count >= 0),
    CHECK (execution_fence >= 0),
    CHECK (duration_ms IS NULL OR duration_ms >= 0),
    CHECK (source_token_estimate IS NULL OR source_token_estimate >= 0),
    CHECK (
        status IN ('succeeded', 'skipped', 'failed', 'dead_lettered', 'cancelled')
        OR (
            envelope_schema_version = 1
            AND recovery_envelope_json IS NOT NULL
            AND length(recovery_envelope_json) > 0
        )
    )
);

INSERT INTO worker_job_runs(
    _rowid,
    job_id,
    stream_name,
    target_backend,
    job_type,
    user_id,
    conversation_id,
    source_message_ids_json,
    status,
    attempt_count,
    source_token_estimate,
    size_bucket,
    queued_at,
    started_at,
    finished_at,
    last_heartbeat_at,
    duration_ms,
    error_class,
    error_message,
    metadata_json,
    user_persona_id,
    platform_id,
    character_id,
    incognito_snapshot,
    remember_across_chats_snapshot,
    remember_across_devices_snapshot,
    temporary_snapshot,
    purge_on_close_snapshot,
    policy_snapshot_json,
    deferred_until,
    transient_defer_count,
    first_deferred_at,
    last_deferred_at,
    lifecycle_epoch,
    lifecycle_cleanup_key
)
SELECT
    legacy._rowid,
    legacy.job_id,
    legacy.stream_name,
    'legacy_terminal',
    legacy.job_type,
    legacy.user_id,
    legacy.conversation_id,
    legacy.source_message_ids_json,
    legacy.status,
    legacy.attempt_count,
    legacy.source_token_estimate,
    legacy.size_bucket,
    legacy.queued_at,
    legacy.started_at,
    legacy.finished_at,
    legacy.last_heartbeat_at,
    legacy.duration_ms,
    legacy.error_class,
    legacy.error_message,
    legacy.metadata_json,
    legacy.user_persona_id,
    legacy.platform_id,
    legacy.character_id,
    legacy.incognito_snapshot,
    legacy.remember_across_chats_snapshot,
    legacy.remember_across_devices_snapshot,
    legacy.temporary_snapshot,
    legacy.purge_on_close_snapshot,
    legacy.policy_snapshot_json,
    legacy.deferred_until,
    legacy.transient_defer_count,
    legacy.first_deferred_at,
    legacy.last_deferred_at,
    lifecycle.lifecycle_epoch,
    lifecycle.lifecycle_cleanup_key
FROM worker_job_runs_legacy AS legacy
JOIN user_lifecycles AS lifecycle
  ON lifecycle.user_id = legacy.user_id;

DROP TABLE worker_job_runs_legacy;
PRAGMA legacy_alter_table = OFF;

CREATE INDEX idx_worker_job_runs_user_status
    ON worker_job_runs(user_id, status, queued_at DESC, job_id ASC);
CREATE INDEX idx_worker_job_runs_conversation_status
    ON worker_job_runs(user_id, conversation_id, status, queued_at DESC, job_id ASC);
CREATE INDEX idx_worker_job_runs_type_finished
    ON worker_job_runs(job_type, size_bucket, finished_at DESC, job_id ASC);
CREATE INDEX idx_worker_job_runs_user_conversation_queued
    ON worker_job_runs(user_id, conversation_id, queued_at DESC, job_id ASC);
CREATE INDEX idx_worker_job_runs_dispatchable
    ON worker_job_runs(status, deferred_until, dispatch_visibility_deadline, queued_at, job_id);
CREATE INDEX idx_worker_job_runs_execution_lease
    ON worker_job_runs(status, execution_lease_expires_at, job_id);
CREATE INDEX idx_worker_job_runs_lifecycle
    ON worker_job_runs(user_id, lifecycle_epoch, status, queued_at, job_id);
