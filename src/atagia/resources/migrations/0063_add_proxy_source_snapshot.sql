-- Bind every in-flight proxy generation to the exact canonical source revision
-- that admitted its immutable request message.

ALTER TABLE proxy_turn_runs
    ADD COLUMN lifecycle_epoch TEXT;

ALTER TABLE proxy_turn_runs
    ADD COLUMN derivation_revision INTEGER
    CHECK (derivation_revision IS NULL OR derivation_revision >= 0);

-- A pre-cutover nonterminal generation has no trustworthy observation point
-- for these coordinates. Fail it closed instead of allowing a post-upgrade
-- owner to resume work against an arbitrarily newer transcript.
UPDATE proxy_turn_runs
SET state = 'final_fingerprint_conflict',
    owner_token = NULL,
    lease_expires_at = NULL,
    error_code = 'proxy_source_snapshot_missing',
    error_message = 'Pre-cutover generation has no canonical source snapshot',
    updated_at = CURRENT_TIMESTAMP
WHERE state = 'generating';

UPDATE proxy_turn_runs
SET state = 'ambiguous_exposed',
    owner_token = NULL,
    lease_expires_at = NULL,
    ambiguous_at = COALESCE(ambiguous_at, CURRENT_TIMESTAMP),
    error_code = 'proxy_source_snapshot_missing',
    error_message = 'Pre-cutover stream has no canonical source snapshot',
    updated_at = CURRENT_TIMESTAMP
WHERE state = 'emission_started';

-- Coordinates on completed rows are diagnostic only; retaining the currently
-- known values makes operational inspection useful without changing replay.
UPDATE proxy_turn_runs
SET lifecycle_epoch = (
        SELECT lifecycle.lifecycle_epoch
        FROM user_lifecycles AS lifecycle
        WHERE lifecycle.user_id = proxy_turn_runs.user_id
    ),
    derivation_revision = (
        SELECT lifecycle.derivation_revision
        FROM user_lifecycles AS lifecycle
        WHERE lifecycle.user_id = proxy_turn_runs.user_id
    )
WHERE state = 'completed';

CREATE INDEX idx_proxy_turn_runs_user_source_snapshot
    ON proxy_turn_runs(
        user_id,
        lifecycle_epoch,
        derivation_revision,
        state,
        pair_id
    );
