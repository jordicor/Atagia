-- Durable, lifecycle-fenced user erasure.  The retained tombstone contains no
-- raw user identifier; the resumable cleanup record is deleted atomically once
-- every target has been verified.

ALTER TABLE deletion_tombstones
    ADD COLUMN erasure_protocol_version INTEGER;

ALTER TABLE deletion_tombstones
    ADD COLUMN erasure_cleanup_state TEXT NOT NULL DEFAULT 'not_applicable'
    CHECK (
        erasure_cleanup_state IN (
            'not_applicable',
            'legacy_unknown',
            'pending',
            'verified'
        )
    );

ALTER TABLE deletion_tombstones
    ADD COLUMN cleanup_verified_at TEXT;

ALTER TABLE deletion_tombstones
    ADD COLUMN cleanup_evidence_manifest_sha256 TEXT;

ALTER TABLE deletion_tombstones
    ADD COLUMN cleanup_evidence_references_json TEXT NOT NULL DEFAULT '[]';

ALTER TABLE deletion_tombstones
    ADD COLUMN erasure_lifecycle_epoch TEXT;

ALTER TABLE deletion_tombstones
    ADD COLUMN legacy_reconciled INTEGER NOT NULL DEFAULT 0
    CHECK (legacy_reconciled IN (0, 1));

ALTER TABLE deletion_tombstones
    ADD COLUMN erasure_row_version INTEGER NOT NULL DEFAULT 0
    CHECK (erasure_row_version >= 0);

ALTER TABLE user_lifecycles
    ADD COLUMN erasure_cleanup_id TEXT;

CREATE UNIQUE INDEX idx_user_lifecycles_erasure_cleanup
    ON user_lifecycles(erasure_cleanup_id)
    WHERE erasure_cleanup_id IS NOT NULL;

-- No pre-cutover marker proves that best-effort external cleanup completed.
UPDATE deletion_tombstones
SET erasure_protocol_version = 0,
    erasure_cleanup_state = 'legacy_unknown',
    cleanup_verified_at = NULL,
    cleanup_evidence_manifest_sha256 = NULL,
    cleanup_evidence_references_json = '[]',
    erasure_lifecycle_epoch = NULL,
    legacy_reconciled = 0,
    erasure_row_version = erasure_row_version + 1
WHERE entity_type = 'user'
  AND deletion_reason = 'right_to_erasure';

CREATE TABLE user_erasure_cleanups (
    cleanup_id TEXT PRIMARY KEY,
    tombstone_id TEXT NOT NULL UNIQUE
        REFERENCES deletion_tombstones(id) ON DELETE RESTRICT,
    cleanup_kind TEXT NOT NULL CHECK (
        cleanup_kind IN ('current', 'legacy_reconciliation')
    ),
    candidate_user_id TEXT NOT NULL,
    user_id_sha256 TEXT NOT NULL,
    lifecycle_epoch TEXT,
    lifecycle_cleanup_key TEXT NOT NULL,
    protocol_version INTEGER NOT NULL CHECK (protocol_version > 0),
    inventory_manifest_sha256 TEXT,
    canonical_deleted_at TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    last_attempt_at TEXT,
    last_error TEXT,
    record_version INTEGER NOT NULL DEFAULT 0 CHECK (record_version >= 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (
        (cleanup_kind = 'current' AND lifecycle_epoch IS NOT NULL)
        OR
        (cleanup_kind = 'legacy_reconciliation' AND lifecycle_epoch IS NULL)
    ),
    CHECK (length(user_id_sha256) = 64),
    CHECK (
        inventory_manifest_sha256 IS NULL
        OR length(inventory_manifest_sha256) = 64
    )
);

CREATE UNIQUE INDEX idx_user_erasure_cleanups_candidate
    ON user_erasure_cleanups(candidate_user_id);

CREATE INDEX idx_user_erasure_cleanups_resume
    ON user_erasure_cleanups(updated_at, cleanup_id);

CREATE TABLE user_erasure_cleanup_targets (
    target_id TEXT PRIMARY KEY,
    cleanup_id TEXT NOT NULL
        REFERENCES user_erasure_cleanups(cleanup_id) ON DELETE CASCADE,
    target_kind TEXT NOT NULL,
    backend_name TEXT NOT NULL DEFAULT '',
    target_key TEXT NOT NULL,
    lifecycle_epoch TEXT,
    checkpoint_state TEXT NOT NULL DEFAULT 'pending' CHECK (
        checkpoint_state IN ('pending', 'verified', 'decommissioned')
    ),
    evidence_sha256 TEXT,
    evidence_reference TEXT,
    verified_at TEXT,
    row_version INTEGER NOT NULL DEFAULT 0 CHECK (row_version >= 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (evidence_sha256 IS NULL OR length(evidence_sha256) = 64),
    UNIQUE(cleanup_id, target_kind, backend_name, target_key)
);

CREATE INDEX idx_user_erasure_cleanup_targets_pending
    ON user_erasure_cleanup_targets(
        cleanup_id,
        checkpoint_state,
        target_kind,
        target_id
    );

-- Exact transient delivery coordinates are retained only while cleanup is
-- pending.  They disappear with the cleanup record after verified purge.
CREATE TABLE user_erasure_revoked_jobs (
    cleanup_id TEXT NOT NULL
        REFERENCES user_erasure_cleanups(cleanup_id) ON DELETE CASCADE,
    job_id TEXT NOT NULL,
    conversation_id TEXT,
    stream_name TEXT NOT NULL,
    target_backend TEXT NOT NULL,
    dispatch_token TEXT,
    prior_status TEXT NOT NULL,
    invalidated_execution_fence INTEGER NOT NULL
        CHECK (invalidated_execution_fence > 0),
    purge_state TEXT NOT NULL CHECK (
        purge_state IN ('pending', 'not_required', 'verified', 'decommissioned')
    ),
    evidence_sha256 TEXT,
    evidence_reference TEXT,
    invalidated_at TEXT NOT NULL,
    purged_at TEXT,
    row_version INTEGER NOT NULL DEFAULT 0 CHECK (row_version >= 0),
    CHECK (evidence_sha256 IS NULL OR length(evidence_sha256) = 64),
    PRIMARY KEY (cleanup_id, job_id)
);

CREATE INDEX idx_user_erasure_revoked_jobs_pending
    ON user_erasure_revoked_jobs(cleanup_id, purge_state, job_id);

-- Old code inserting a new erasure marker cannot accidentally bless it.  Such
-- a marker has no protocol evidence and is conservatively treated as legacy.
CREATE TRIGGER deletion_tombstone_erasure_legacy_ai
AFTER INSERT ON deletion_tombstones
WHEN NEW.entity_type = 'user'
 AND NEW.deletion_reason = 'right_to_erasure'
 AND NEW.erasure_cleanup_state = 'not_applicable'
BEGIN
    UPDATE deletion_tombstones
    SET erasure_protocol_version = 0,
        erasure_cleanup_state = 'legacy_unknown',
        cleanup_verified_at = NULL,
        cleanup_evidence_manifest_sha256 = NULL,
        cleanup_evidence_references_json = '[]',
        erasure_lifecycle_epoch = NULL,
        legacy_reconciled = 0,
        erasure_row_version = erasure_row_version + 1
    WHERE id = NEW.id;
END;

-- Retention is deliberately fail-closed at the schema boundary.  RAISE(IGNORE)
-- skips only an ineligible erasure marker while allowing unrelated tombstones
-- in the same maintenance DELETE to be retired.
CREATE TRIGGER deletion_tombstone_erasure_retention_bd
BEFORE DELETE ON deletion_tombstones
WHEN OLD.entity_type = 'user'
 AND OLD.deletion_reason = 'right_to_erasure'
 AND (
    OLD.erasure_protocol_version IS NOT 1
    OR OLD.erasure_cleanup_state IS NOT 'verified'
    OR OLD.cleanup_verified_at IS NULL
    OR OLD.cleanup_evidence_manifest_sha256 IS NULL
    OR (
        OLD.erasure_lifecycle_epoch IS NULL
        AND OLD.legacy_reconciled != 1
    )
    OR EXISTS (
        SELECT 1
        FROM user_erasure_cleanups AS cleanup
        WHERE cleanup.tombstone_id = OLD.id
    )
    OR EXISTS (
        SELECT 1
        FROM user_lifecycles AS lifecycle
        WHERE OLD.erasure_lifecycle_epoch IS NOT NULL
          AND lifecycle.lifecycle_epoch = OLD.erasure_lifecycle_epoch
    )
    OR EXISTS (
        SELECT 1
        FROM worker_job_runs AS job
        WHERE OLD.erasure_lifecycle_epoch IS NOT NULL
          AND job.lifecycle_epoch = OLD.erasure_lifecycle_epoch
          AND (
              job.status IN (
                  'queued',
                  'awaiting_claim',
                  'running',
                  'retrying',
                  'deferred'
              )
              OR job.recovery_envelope_json IS NOT NULL
          )
    )
 )
BEGIN
    SELECT RAISE(IGNORE);
END;
