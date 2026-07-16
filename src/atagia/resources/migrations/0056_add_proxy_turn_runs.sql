CREATE TABLE proxy_turn_runs (
    pair_id TEXT PRIMARY KEY,
    request_message_id TEXT NOT NULL UNIQUE,
    response_message_id TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    request_message_role TEXT NOT NULL CHECK (
        request_message_role IN ('user', 'tool')
    ),
    request_source_seq INTEGER CHECK (request_source_seq IS NULL OR request_source_seq > 0),
    response_source_seq INTEGER CHECK (response_source_seq IS NULL OR response_source_seq > 0),
    state TEXT NOT NULL CHECK (
        state IN (
            'generating',
            'emission_started',
            'completed',
            'ambiguous_exposed',
            'final_fingerprint_conflict'
        )
    ),
    client_fingerprint_version INTEGER NOT NULL CHECK (
        client_fingerprint_version = 1
    ),
    client_request_fingerprint TEXT NOT NULL,
    final_fingerprint_version INTEGER CHECK (
        final_fingerprint_version IS NULL OR final_fingerprint_version = 1
    ),
    final_provider_fingerprint TEXT,
    owner_token TEXT,
    owner_fence INTEGER NOT NULL DEFAULT 1 CHECK (owner_fence >= 1),
    lease_expires_at TEXT,
    emission_started_at TEXT,
    response_linked_at TEXT,
    durable_job_ids_json TEXT NOT NULL DEFAULT '[]',
    completed_at TEXT,
    ambiguous_at TEXT,
    error_code TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (request_message_id <> response_message_id),
    CHECK (
        request_source_seq IS NULL
        OR response_source_seq IS NULL
        OR request_source_seq < response_source_seq
    ),
    CHECK (
        (final_fingerprint_version IS NULL AND final_provider_fingerprint IS NULL)
        OR
        (final_fingerprint_version = 1 AND final_provider_fingerprint IS NOT NULL)
    ),
    CHECK (
        state <> 'completed'
        OR (
            completed_at IS NOT NULL
            AND response_linked_at IS NOT NULL
            AND request_source_seq IS NOT NULL
            AND response_source_seq IS NOT NULL
            AND final_provider_fingerprint IS NOT NULL
            AND json_valid(durable_job_ids_json)
            AND json_type(durable_job_ids_json) = 'array'
        )
    ),
    CHECK (
        state NOT IN ('emission_started', 'ambiguous_exposed')
        OR emission_started_at IS NOT NULL
    )
);

CREATE INDEX idx_proxy_turn_runs_owner_lease
    ON proxy_turn_runs(state, lease_expires_at, pair_id);
CREATE INDEX idx_proxy_turn_runs_user_conversation
    ON proxy_turn_runs(user_id, conversation_id, created_at, pair_id);

CREATE TABLE proxy_message_id_claims (
    message_id TEXT PRIMARY KEY,
    pair_id TEXT NOT NULL REFERENCES proxy_turn_runs(pair_id) ON DELETE CASCADE,
    pair_role TEXT NOT NULL CHECK (pair_role IN ('request', 'response')),
    message_role TEXT NOT NULL CHECK (
        message_role IN ('user', 'tool', 'assistant')
    ),
    counterpart_message_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    claim_token TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(pair_id, pair_role),
    CHECK (message_id <> counterpart_message_id),
    CHECK (
        (pair_role = 'request' AND message_role IN ('user', 'tool'))
        OR (pair_role = 'response' AND message_role = 'assistant')
    )
);

CREATE INDEX idx_proxy_message_id_claims_pair
    ON proxy_message_id_claims(pair_id, pair_role);
CREATE INDEX idx_proxy_message_id_claims_namespace
    ON proxy_message_id_claims(user_id, conversation_id, message_id);

-- A previous experimental build may already have reciprocal version-1 proxy
-- transcript metadata. Backfill only pairs whose two immutable message rows
-- prove the same pair in both directions. Any other pre-cutover message ID is
-- intentionally left unclaimed and therefore remains occupied for proxy reuse.
INSERT INTO proxy_turn_runs(
    pair_id,
    request_message_id,
    response_message_id,
    user_id,
    conversation_id,
    request_message_role,
    request_source_seq,
    response_source_seq,
    state,
    client_fingerprint_version,
    client_request_fingerprint,
    final_fingerprint_version,
    final_provider_fingerprint,
    owner_fence,
    response_linked_at,
    durable_job_ids_json,
    completed_at,
    created_at,
    updated_at
)
SELECT
    json_extract(response.metadata_json, '$.atagia_proxy_transcript.pair_id'),
    request.id,
    response.id,
    conversation.user_id,
    response.conversation_id,
    request.role,
    request.seq,
    response.seq,
    'completed',
    1,
    json_extract(
        response.metadata_json,
        '$.atagia_proxy_transcript.client_request_fingerprint'
    ),
    1,
    json_extract(
        response.metadata_json,
        '$.atagia_proxy_transcript.final_provider_fingerprint'
    ),
    1,
    response.created_at,
    '[]',
    response.created_at,
    request.created_at,
    response.created_at
FROM messages AS response
JOIN messages AS request
  ON request.id = json_extract(
      response.metadata_json,
      '$.atagia_proxy_transcript.request_message_id'
  )
 AND request.conversation_id = response.conversation_id
JOIN conversations AS conversation
  ON conversation.id = response.conversation_id
WHERE response.role = 'assistant'
  AND request.role IN ('user', 'tool')
  AND json_valid(response.metadata_json)
  AND json_valid(request.metadata_json)
  AND json_extract(
      response.metadata_json,
      '$.atagia_proxy_transcript.schema_version'
  ) = 1
  AND json_extract(
      request.metadata_json,
      '$.atagia_proxy_transcript.schema_version'
  ) = 1
  AND json_extract(
      request.metadata_json,
      '$.atagia_proxy_transcript.expected_response_message_id'
  ) = response.id
  AND json_extract(
      request.metadata_json,
      '$.atagia_proxy_transcript.pair_id'
  ) = json_extract(
      response.metadata_json,
      '$.atagia_proxy_transcript.pair_id'
  )
  AND json_extract(
      response.metadata_json,
      '$.atagia_proxy_transcript.client_request_fingerprint'
  ) IS NOT NULL
  AND json_extract(
      response.metadata_json,
      '$.atagia_proxy_transcript.final_provider_fingerprint'
  ) IS NOT NULL;

INSERT INTO proxy_message_id_claims(
    message_id,
    pair_id,
    pair_role,
    message_role,
    counterpart_message_id,
    user_id,
    conversation_id,
    claim_token,
    created_at
)
SELECT
    request_message_id,
    pair_id,
    'request',
    request_message_role,
    response_message_id,
    user_id,
    conversation_id,
    'legacy-paired:' || pair_id,
    created_at
FROM proxy_turn_runs;

INSERT INTO proxy_message_id_claims(
    message_id,
    pair_id,
    pair_role,
    message_role,
    counterpart_message_id,
    user_id,
    conversation_id,
    claim_token,
    created_at
)
SELECT
    response_message_id,
    pair_id,
    'response',
    'assistant',
    request_message_id,
    user_id,
    conversation_id,
    'legacy-paired:' || pair_id,
    created_at
FROM proxy_turn_runs;

-- Minimal pair identity and conflict evidence live exactly as long as at least
-- one paired message. Deleting the final message removes the run and its two
-- claims in the same transaction through the claims' ON DELETE CASCADE.
CREATE TRIGGER proxy_turn_runs_delete_after_last_message
AFTER DELETE ON messages
BEGIN
    DELETE FROM proxy_turn_runs
    WHERE (request_message_id = OLD.id OR response_message_id = OLD.id)
      AND NOT EXISTS (
          SELECT 1
          FROM messages
          WHERE messages.id = proxy_turn_runs.request_message_id
             OR messages.id = proxy_turn_runs.response_message_id
      );
END;
