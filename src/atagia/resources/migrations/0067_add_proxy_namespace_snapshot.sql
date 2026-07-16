-- Bind every in-flight proxy generation to the exact conversation namespace
-- that admitted its immutable request message.

ALTER TABLE proxy_turn_runs
    ADD COLUMN namespace_snapshot_json TEXT
    CHECK (
        namespace_snapshot_json IS NULL
        OR (
            json_valid(namespace_snapshot_json)
            AND json_type(namespace_snapshot_json) = 'object'
        )
    );

-- A pre-cutover nonterminal generation has no trustworthy admission snapshot.
-- Fail it closed; completed rows retain replay semantics independently of the
-- conversation's current dynamic namespace.
UPDATE proxy_turn_runs
SET state = 'final_fingerprint_conflict',
    owner_token = NULL,
    lease_expires_at = NULL,
    error_code = 'proxy_namespace_snapshot_missing',
    error_message = 'Pre-cutover generation has no conversation namespace snapshot',
    updated_at = CURRENT_TIMESTAMP
WHERE state = 'generating'
  AND namespace_snapshot_json IS NULL;

UPDATE proxy_turn_runs
SET state = 'ambiguous_exposed',
    owner_token = NULL,
    lease_expires_at = NULL,
    ambiguous_at = COALESCE(ambiguous_at, CURRENT_TIMESTAMP),
    error_code = 'proxy_namespace_snapshot_missing',
    error_message = 'Pre-cutover stream has no conversation namespace snapshot',
    updated_at = CURRENT_TIMESTAMP
WHERE state = 'emission_started'
  AND namespace_snapshot_json IS NULL;

CREATE INDEX idx_proxy_turn_runs_namespace_snapshot_state
    ON proxy_turn_runs(user_id, conversation_id, state, pair_id)
    WHERE namespace_snapshot_json IS NOT NULL;
