-- SQLite-owned cache identity and exact initial-context-package source revisions.

ALTER TABLE user_lifecycles
    ADD COLUMN source_revision INTEGER NOT NULL DEFAULT 0
    CHECK (source_revision >= 0);

ALTER TABLE user_lifecycles
    ADD COLUMN icp_refresh_generation INTEGER NOT NULL DEFAULT 0
    CHECK (icp_refresh_generation >= 0);

CREATE TABLE conversation_lifecycles (
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    lifecycle_epoch TEXT NOT NULL UNIQUE,
    source_revision INTEGER NOT NULL DEFAULT 0 CHECK (source_revision >= 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, conversation_id)
);

INSERT INTO conversation_lifecycles(
    user_id,
    conversation_id,
    lifecycle_epoch,
    source_revision,
    created_at,
    updated_at
)
SELECT
    user_id,
    id,
    'cle_' || lower(hex(randomblob(16))),
    0,
    created_at,
    updated_at
FROM conversations;

CREATE INDEX idx_conversation_lifecycles_epoch
    ON conversation_lifecycles(user_id, lifecycle_epoch, conversation_id);

CREATE TRIGGER conversation_lifecycle_owner_bi
BEFORE INSERT ON conversation_lifecycles
BEGIN
    SELECT RAISE(ABORT, 'conversation lifecycle must belong to user')
    WHERE NOT EXISTS (
        SELECT 1
        FROM conversations
        WHERE conversations.id = NEW.conversation_id
          AND conversations.user_id = NEW.user_id
    );
END;

CREATE TRIGGER conversation_lifecycle_create_ai
AFTER INSERT ON conversations
BEGIN
    INSERT INTO conversation_lifecycles(
        user_id,
        conversation_id,
        lifecycle_epoch,
        source_revision,
        created_at,
        updated_at
    )
    VALUES (
        NEW.user_id,
        NEW.id,
        'cle_' || lower(hex(randomblob(16))),
        0,
        NEW.created_at,
        NEW.updated_at
    );
END;

ALTER TABLE initial_context_packages
    ADD COLUMN source_user_lifecycle_epoch TEXT;

ALTER TABLE initial_context_packages
    ADD COLUMN source_user_revision INTEGER;

ALTER TABLE initial_context_packages
    ADD COLUMN source_conversation_lifecycle_epoch TEXT;

ALTER TABLE initial_context_packages
    ADD COLUMN source_conversation_revision INTEGER;

ALTER TABLE initial_context_packages
    ADD COLUMN package_row_version INTEGER NOT NULL DEFAULT 0
    CHECK (package_row_version >= 0);

ALTER TABLE initial_context_packages
    ADD COLUMN active_build_attempt_id TEXT;

ALTER TABLE initial_context_packages
    ADD COLUMN refresh_generation INTEGER NOT NULL DEFAULT 0
    CHECK (refresh_generation >= 0);

ALTER TABLE initial_context_packages
    ADD COLUMN last_refresh_request_job_id TEXT;

-- Pre-cutover packages never captured exact revisions and cannot remain active.
UPDATE initial_context_packages
SET build_status = 'stale',
    source_user_lifecycle_epoch = NULL,
    source_user_revision = NULL,
    source_conversation_lifecycle_epoch = NULL,
    source_conversation_revision = NULL,
    package_row_version = package_row_version + 1,
    active_build_attempt_id = NULL;

CREATE TABLE initial_context_package_build_attempts (
    attempt_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    conversation_id TEXT,
    package_key_hash TEXT NOT NULL,
    refresh_generation INTEGER NOT NULL CHECK (refresh_generation > 0),
    expected_package_row_version INTEGER,
    source_user_lifecycle_epoch TEXT NOT NULL,
    source_user_revision INTEGER NOT NULL CHECK (source_user_revision >= 0),
    source_conversation_lifecycle_epoch TEXT,
    source_conversation_revision INTEGER,
    status TEXT NOT NULL CHECK (
        status IN (
            'building',
            'activated',
            'source_changed',
            'superseded',
            'failed'
        )
    ),
    refresh_request_job_id TEXT,
    created_at TEXT NOT NULL,
    finished_at TEXT,
    diagnostics_json TEXT NOT NULL DEFAULT '{}',
    CHECK (
        (conversation_id IS NULL
         AND source_conversation_lifecycle_epoch IS NULL
         AND source_conversation_revision IS NULL)
        OR
        (conversation_id IS NOT NULL
         AND source_conversation_lifecycle_epoch IS NOT NULL
         AND source_conversation_revision IS NOT NULL)
    )
);

CREATE INDEX idx_icp_build_attempts_user_status
    ON initial_context_package_build_attempts(
        user_id,
        status,
        refresh_generation,
        attempt_id
    );

CREATE INDEX idx_icp_build_attempts_package_generation
    ON initial_context_package_build_attempts(
        package_key_hash,
        refresh_generation DESC,
        attempt_id
    );

CREATE UNIQUE INDEX idx_worker_job_runs_active_icp_refresh_dedupe
    ON worker_job_runs(
        user_id,
        json_extract(recovery_envelope_json, '$.payload.refresh_dedupe_key')
    )
    WHERE job_type = 'refresh_initial_context_package'
      AND status IN (
          'queued',
          'awaiting_claim',
          'running',
          'retrying',
          'deferred'
      )
      AND recovery_envelope_json IS NOT NULL;

CREATE TRIGGER icp_source_users_au
AFTER UPDATE ON users
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = NEW.id;
END;

CREATE TRIGGER icp_source_users_bd
BEFORE DELETE ON users
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = OLD.id;
END;

CREATE TRIGGER icp_source_conversations_au
AFTER UPDATE ON conversations
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id IN (OLD.user_id, NEW.user_id)
      AND (
          OLD.user_id IS NOT NEW.user_id
          OR OLD.workspace_id IS NOT NEW.workspace_id
          OR OLD.assistant_mode_id IS NOT NEW.assistant_mode_id
          OR OLD.status IS NOT NEW.status
          OR OLD.temporary IS NOT NEW.temporary
          OR OLD.temporary_ttl_seconds IS NOT NEW.temporary_ttl_seconds
          OR OLD.purge_on_close IS NOT NEW.purge_on_close
          OR OLD.isolated_mode IS NOT NEW.isolated_mode
          OR OLD.user_persona_id IS NOT NEW.user_persona_id
          OR OLD.platform_id IS NOT NEW.platform_id
          OR OLD.character_id IS NOT NEW.character_id
          OR OLD.mode IS NOT NEW.mode
          OR OLD.incognito IS NOT NEW.incognito
          OR OLD.active_presence_id IS NOT NEW.active_presence_id
          OR OLD.active_space_id IS NOT NEW.active_space_id
          OR OLD.active_mind_id IS NOT NEW.active_mind_id
          OR OLD.mind_topology IS NOT NEW.mind_topology
          OR OLD.active_embodiment_id IS NOT NEW.active_embodiment_id
          OR OLD.active_realm_id IS NOT NEW.active_realm_id
      );

    -- Ownership moves create a new immutable conversation lifecycle. Removing
    -- the OLD row makes every package captured for the prior owner fail closed;
    -- the fresh random epoch prevents an old builder from activating for NEW.
    DELETE FROM conversation_lifecycles
    WHERE OLD.user_id IS NOT NEW.user_id
      AND user_id = OLD.user_id
      AND conversation_id = OLD.id;

    INSERT INTO conversation_lifecycles(
        user_id,
        conversation_id,
        lifecycle_epoch,
        source_revision,
        created_at,
        updated_at
    )
    SELECT
        NEW.user_id,
        NEW.id,
        'cle_' || lower(hex(randomblob(16))),
        0,
        NEW.created_at,
        NEW.updated_at
    WHERE OLD.user_id IS NOT NEW.user_id;

    UPDATE conversation_lifecycles
    SET source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE (user_id = OLD.user_id AND conversation_id = OLD.id)
       OR (user_id = NEW.user_id AND conversation_id = NEW.id);
END;

CREATE TRIGGER icp_source_messages_ai
AFTER INSERT ON messages
BEGIN
    UPDATE conversation_lifecycles
    SET source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE conversation_id = NEW.conversation_id;
END;

CREATE TRIGGER icp_source_messages_au
AFTER UPDATE ON messages
BEGIN
    UPDATE conversation_lifecycles
    SET source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE conversation_id IN (OLD.conversation_id, NEW.conversation_id);
END;

CREATE TRIGGER icp_source_messages_bd
BEFORE DELETE ON messages
BEGIN
    UPDATE conversation_lifecycles
    SET source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE conversation_id = OLD.conversation_id;
END;

CREATE TRIGGER icp_source_belief_versions_ai
AFTER INSERT ON belief_versions
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = (
        SELECT user_id FROM memory_objects WHERE id = NEW.belief_id
    );
END;

CREATE TRIGGER icp_source_belief_versions_au
AFTER UPDATE ON belief_versions
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id IN (
        SELECT user_id FROM memory_objects
        WHERE id IN (OLD.belief_id, NEW.belief_id)
    );
END;

CREATE TRIGGER icp_source_belief_versions_bd
BEFORE DELETE ON belief_versions
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = (
        SELECT user_id FROM memory_objects WHERE id = OLD.belief_id
    );
END;

CREATE TRIGGER icp_source_conversation_activity_stats_ai
AFTER INSERT ON conversation_activity_stats
BEGIN
    UPDATE conversation_lifecycles
    SET source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = NEW.user_id
      AND conversation_id = NEW.conversation_id;
END;

CREATE TRIGGER icp_source_conversation_activity_stats_au
AFTER UPDATE ON conversation_activity_stats
BEGIN
    UPDATE conversation_lifecycles
    SET source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE (user_id = OLD.user_id AND conversation_id = OLD.conversation_id)
       OR (user_id = NEW.user_id AND conversation_id = NEW.conversation_id);
END;

CREATE TRIGGER icp_source_conversation_activity_stats_bd
BEFORE DELETE ON conversation_activity_stats
BEGIN
    UPDATE conversation_lifecycles
    SET source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = OLD.user_id
      AND conversation_id = OLD.conversation_id;
END;

CREATE TRIGGER icp_source_memory_objects_ai
AFTER INSERT ON memory_objects
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = NEW.user_id;
END;

CREATE TRIGGER icp_source_memory_objects_au
AFTER UPDATE ON memory_objects
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id IN (OLD.user_id, NEW.user_id);
END;

CREATE TRIGGER icp_source_memory_objects_bd
BEFORE DELETE ON memory_objects
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = OLD.user_id;
END;

CREATE TRIGGER icp_source_memory_links_ai
AFTER INSERT ON memory_links
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = NEW.user_id;
END;

CREATE TRIGGER icp_source_memory_links_au
AFTER UPDATE ON memory_links
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id IN (OLD.user_id, NEW.user_id);
END;

CREATE TRIGGER icp_source_memory_links_bd
BEFORE DELETE ON memory_links
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = OLD.user_id;
END;

CREATE TRIGGER icp_source_memory_retrieval_surfaces_ai
AFTER INSERT ON memory_retrieval_surfaces
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = NEW.user_id;
END;

CREATE TRIGGER icp_source_memory_retrieval_surfaces_au
AFTER UPDATE ON memory_retrieval_surfaces
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id IN (OLD.user_id, NEW.user_id);
END;

CREATE TRIGGER icp_source_memory_retrieval_surfaces_bd
BEFORE DELETE ON memory_retrieval_surfaces
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = OLD.user_id;
END;

CREATE TRIGGER icp_source_contract_dimensions_current_ai
AFTER INSERT ON contract_dimensions_current
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = NEW.user_id;
END;

CREATE TRIGGER icp_source_contract_dimensions_current_au
AFTER UPDATE ON contract_dimensions_current
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id IN (OLD.user_id, NEW.user_id);
END;

CREATE TRIGGER icp_source_contract_dimensions_current_bd
BEFORE DELETE ON contract_dimensions_current
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = OLD.user_id;
END;

CREATE TRIGGER icp_source_consequence_chains_ai
AFTER INSERT ON consequence_chains
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = NEW.user_id;
END;

CREATE TRIGGER icp_source_consequence_chains_au
AFTER UPDATE ON consequence_chains
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id IN (OLD.user_id, NEW.user_id);
END;

CREATE TRIGGER icp_source_consequence_chains_bd
BEFORE DELETE ON consequence_chains
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = OLD.user_id;
END;

CREATE TRIGGER icp_source_summary_views_ai
AFTER INSERT ON summary_views
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = NEW.user_id;
END;

CREATE TRIGGER icp_source_summary_views_au
AFTER UPDATE ON summary_views
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id IN (OLD.user_id, NEW.user_id);
END;

CREATE TRIGGER icp_source_summary_views_bd
BEFORE DELETE ON summary_views
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = OLD.user_id;
END;

CREATE TRIGGER icp_source_user_communication_profiles_ai
AFTER INSERT ON user_communication_profiles
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = NEW.user_id;
END;

CREATE TRIGGER icp_source_user_communication_profiles_au
AFTER UPDATE ON user_communication_profiles
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id IN (OLD.user_id, NEW.user_id);
END;

CREATE TRIGGER icp_source_user_communication_profiles_bd
BEFORE DELETE ON user_communication_profiles
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = OLD.user_id;
END;

CREATE TRIGGER icp_source_memory_consent_profile_ai
AFTER INSERT ON memory_consent_profile
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = NEW.user_id;
END;

CREATE TRIGGER icp_source_memory_consent_profile_au
AFTER UPDATE ON memory_consent_profile
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id IN (OLD.user_id, NEW.user_id);
END;

CREATE TRIGGER icp_source_memory_consent_profile_bd
BEFORE DELETE ON memory_consent_profile
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = OLD.user_id;
END;

CREATE TRIGGER icp_source_pending_memory_confirmations_ai
AFTER INSERT ON pending_memory_confirmations
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = NEW.user_id;
END;

CREATE TRIGGER icp_source_pending_memory_confirmations_au
AFTER UPDATE ON pending_memory_confirmations
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id IN (OLD.user_id, NEW.user_id);
END;

CREATE TRIGGER icp_source_pending_memory_confirmations_bd
BEFORE DELETE ON pending_memory_confirmations
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = OLD.user_id;
END;

CREATE TRIGGER icp_source_artifacts_ai
AFTER INSERT ON artifacts
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = NEW.user_id;
END;

CREATE TRIGGER icp_source_artifacts_au
AFTER UPDATE ON artifacts
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id IN (OLD.user_id, NEW.user_id);
END;

CREATE TRIGGER icp_source_artifacts_bd
BEFORE DELETE ON artifacts
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = OLD.user_id;
END;

CREATE TRIGGER icp_source_artifact_chunks_ai
AFTER INSERT ON artifact_chunks
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = NEW.user_id;
END;

CREATE TRIGGER icp_source_artifact_chunks_au
AFTER UPDATE ON artifact_chunks
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id IN (OLD.user_id, NEW.user_id);
END;

CREATE TRIGGER icp_source_artifact_chunks_bd
BEFORE DELETE ON artifact_chunks
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = OLD.user_id;
END;

CREATE TRIGGER icp_source_artifact_payload_blobs_ai
AFTER INSERT ON artifact_payload_blobs
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = NEW.user_id;
END;

CREATE TRIGGER icp_source_artifact_payload_blobs_au
AFTER UPDATE ON artifact_payload_blobs
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id IN (OLD.user_id, NEW.user_id);
END;

CREATE TRIGGER icp_source_artifact_payload_blobs_bd
BEFORE DELETE ON artifact_payload_blobs
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = OLD.user_id;
END;

CREATE TRIGGER icp_source_artifact_links_ai
AFTER INSERT ON artifact_links
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = NEW.user_id;
END;

CREATE TRIGGER icp_source_artifact_links_au
AFTER UPDATE ON artifact_links
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id IN (OLD.user_id, NEW.user_id);
END;

CREATE TRIGGER icp_source_artifact_links_bd
BEFORE DELETE ON artifact_links
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = OLD.user_id;
END;

CREATE TRIGGER icp_source_verbatim_pins_ai
AFTER INSERT ON verbatim_pins
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = NEW.user_id;
END;

CREATE TRIGGER icp_source_verbatim_pins_au
AFTER UPDATE ON verbatim_pins
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id IN (OLD.user_id, NEW.user_id);
END;

CREATE TRIGGER icp_source_verbatim_pins_bd
BEFORE DELETE ON verbatim_pins
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = OLD.user_id;
END;

CREATE TRIGGER icp_source_memory_support_edges_ai
AFTER INSERT ON memory_support_edges
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = NEW.user_id;
END;

CREATE TRIGGER icp_source_memory_support_edges_au
AFTER UPDATE ON memory_support_edges
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id IN (OLD.user_id, NEW.user_id);
END;

CREATE TRIGGER icp_source_memory_support_edges_bd
BEFORE DELETE ON memory_support_edges
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = OLD.user_id;
END;

CREATE TRIGGER icp_source_memory_evidence_spans_ai
AFTER INSERT ON memory_evidence_spans
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = NEW.user_id;
END;

CREATE TRIGGER icp_source_memory_evidence_spans_au
AFTER UPDATE ON memory_evidence_spans
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id IN (OLD.user_id, NEW.user_id);
END;

CREATE TRIGGER icp_source_memory_evidence_spans_bd
BEFORE DELETE ON memory_evidence_spans
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = OLD.user_id;
END;

CREATE TRIGGER icp_source_graph_entities_ai
AFTER INSERT ON graph_entities
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = NEW.user_id;
END;

CREATE TRIGGER icp_source_graph_entities_au
AFTER UPDATE ON graph_entities
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id IN (OLD.user_id, NEW.user_id);
END;

CREATE TRIGGER icp_source_graph_entities_bd
BEFORE DELETE ON graph_entities
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = OLD.user_id;
END;

CREATE TRIGGER icp_source_graph_entity_mentions_ai
AFTER INSERT ON graph_entity_mentions
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = NEW.user_id;
END;

CREATE TRIGGER icp_source_graph_entity_mentions_au
AFTER UPDATE ON graph_entity_mentions
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id IN (OLD.user_id, NEW.user_id);
END;

CREATE TRIGGER icp_source_graph_entity_mentions_bd
BEFORE DELETE ON graph_entity_mentions
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = OLD.user_id;
END;

CREATE TRIGGER icp_source_graph_relationships_ai
AFTER INSERT ON graph_relationships
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = NEW.user_id;
END;

CREATE TRIGGER icp_source_graph_relationships_au
AFTER UPDATE ON graph_relationships
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id IN (OLD.user_id, NEW.user_id);
END;

CREATE TRIGGER icp_source_graph_relationships_bd
BEFORE DELETE ON graph_relationships
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = OLD.user_id;
END;

CREATE TRIGGER icp_source_graph_relationship_sources_ai
AFTER INSERT ON graph_relationship_sources
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = NEW.user_id;
END;

CREATE TRIGGER icp_source_graph_relationship_sources_au
AFTER UPDATE ON graph_relationship_sources
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id IN (OLD.user_id, NEW.user_id);
END;

CREATE TRIGGER icp_source_graph_relationship_sources_bd
BEFORE DELETE ON graph_relationship_sources
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = OLD.user_id;
END;

CREATE TRIGGER icp_source_memory_object_subjects_ai
AFTER INSERT ON memory_object_subjects
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = NEW.owner_user_id;
END;

CREATE TRIGGER icp_source_memory_object_subjects_au
AFTER UPDATE ON memory_object_subjects
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id IN (OLD.owner_user_id, NEW.owner_user_id);
END;

CREATE TRIGGER icp_source_memory_object_subjects_bd
BEFORE DELETE ON memory_object_subjects
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = OLD.owner_user_id;
END;

CREATE TRIGGER icp_source_presences_ai
AFTER INSERT ON presences
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = NEW.owner_user_id;
END;

CREATE TRIGGER icp_source_presences_au
AFTER UPDATE ON presences
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id IN (OLD.owner_user_id, NEW.owner_user_id);
END;

CREATE TRIGGER icp_source_presences_bd
BEFORE DELETE ON presences
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = OLD.owner_user_id;
END;

CREATE TRIGGER icp_source_spaces_ai
AFTER INSERT ON spaces
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = NEW.owner_user_id;
END;

CREATE TRIGGER icp_source_spaces_au
AFTER UPDATE ON spaces
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id IN (OLD.owner_user_id, NEW.owner_user_id);
END;

CREATE TRIGGER icp_source_spaces_bd
BEFORE DELETE ON spaces
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = OLD.owner_user_id;
END;

CREATE TRIGGER icp_source_minds_ai
AFTER INSERT ON minds
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = NEW.owner_user_id;
END;

CREATE TRIGGER icp_source_minds_au
AFTER UPDATE ON minds
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id IN (OLD.owner_user_id, NEW.owner_user_id);
END;

CREATE TRIGGER icp_source_minds_bd
BEFORE DELETE ON minds
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = OLD.owner_user_id;
END;

CREATE TRIGGER icp_source_overseer_grants_ai
AFTER INSERT ON overseer_grants
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = NEW.owner_user_id;
END;

CREATE TRIGGER icp_source_overseer_grants_au
AFTER UPDATE ON overseer_grants
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id IN (OLD.owner_user_id, NEW.owner_user_id);
END;

CREATE TRIGGER icp_source_overseer_grants_bd
BEFORE DELETE ON overseer_grants
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = OLD.owner_user_id;
END;

CREATE TRIGGER icp_source_embodiments_ai
AFTER INSERT ON embodiments
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = NEW.owner_user_id;
END;

CREATE TRIGGER icp_source_embodiments_au
AFTER UPDATE ON embodiments
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id IN (OLD.owner_user_id, NEW.owner_user_id);
END;

CREATE TRIGGER icp_source_embodiments_bd
BEFORE DELETE ON embodiments
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = OLD.owner_user_id;
END;

CREATE TRIGGER icp_source_realms_ai
AFTER INSERT ON realms
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = NEW.owner_user_id;
END;

CREATE TRIGGER icp_source_realms_au
AFTER UPDATE ON realms
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id IN (OLD.owner_user_id, NEW.owner_user_id);
END;

CREATE TRIGGER icp_source_realms_bd
BEFORE DELETE ON realms
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = OLD.owner_user_id;
END;

CREATE TRIGGER icp_source_realm_bridges_ai
AFTER INSERT ON realm_bridges
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = NEW.owner_user_id;
END;

CREATE TRIGGER icp_source_realm_bridges_au
AFTER UPDATE ON realm_bridges
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id IN (OLD.owner_user_id, NEW.owner_user_id);
END;

CREATE TRIGGER icp_source_realm_bridges_bd
BEFORE DELETE ON realm_bridges
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = OLD.owner_user_id;
END;

CREATE TRIGGER icp_source_conversation_topics_ai
AFTER INSERT ON conversation_topics
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = NEW.user_id;

    UPDATE conversation_lifecycles
    SET source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = NEW.user_id
      AND conversation_id = NEW.conversation_id;
END;

CREATE TRIGGER icp_source_conversation_topics_au
AFTER UPDATE ON conversation_topics
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id IN (OLD.user_id, NEW.user_id);

    UPDATE conversation_lifecycles
    SET source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE (user_id = OLD.user_id AND conversation_id = OLD.conversation_id)
       OR (user_id = NEW.user_id AND conversation_id = NEW.conversation_id);
END;

CREATE TRIGGER icp_source_conversation_topics_bd
BEFORE DELETE ON conversation_topics
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = OLD.user_id;

    UPDATE conversation_lifecycles
    SET source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = OLD.user_id
      AND conversation_id = OLD.conversation_id;
END;

CREATE TRIGGER icp_source_conversation_topic_events_ai
AFTER INSERT ON conversation_topic_events
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = NEW.user_id;

    UPDATE conversation_lifecycles
    SET source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = NEW.user_id
      AND conversation_id = NEW.conversation_id;
END;

CREATE TRIGGER icp_source_conversation_topic_events_au
AFTER UPDATE ON conversation_topic_events
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id IN (OLD.user_id, NEW.user_id);

    UPDATE conversation_lifecycles
    SET source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE (user_id = OLD.user_id AND conversation_id = OLD.conversation_id)
       OR (user_id = NEW.user_id AND conversation_id = NEW.conversation_id);
END;

CREATE TRIGGER icp_source_conversation_topic_events_bd
BEFORE DELETE ON conversation_topic_events
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = OLD.user_id;

    UPDATE conversation_lifecycles
    SET source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = OLD.user_id
      AND conversation_id = OLD.conversation_id;
END;

CREATE TRIGGER icp_source_conversation_topic_sources_ai
AFTER INSERT ON conversation_topic_sources
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = NEW.user_id;

    UPDATE conversation_lifecycles
    SET source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = NEW.user_id
      AND conversation_id = (
          SELECT conversation_id
          FROM conversation_topics
          WHERE user_id = NEW.user_id
            AND id = NEW.topic_id
      );
END;

CREATE TRIGGER icp_source_conversation_topic_sources_au
AFTER UPDATE ON conversation_topic_sources
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id IN (OLD.user_id, NEW.user_id);

    UPDATE conversation_lifecycles
    SET source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE (user_id = OLD.user_id AND conversation_id = (
               SELECT conversation_id
               FROM conversation_topics
               WHERE user_id = OLD.user_id AND id = OLD.topic_id
           ))
       OR (user_id = NEW.user_id AND conversation_id = (
               SELECT conversation_id
               FROM conversation_topics
               WHERE user_id = NEW.user_id AND id = NEW.topic_id
           ));
END;

CREATE TRIGGER icp_source_conversation_topic_sources_bd
BEFORE DELETE ON conversation_topic_sources
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = OLD.user_id;

    UPDATE conversation_lifecycles
    SET source_revision = source_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = OLD.user_id
      AND conversation_id = (
          SELECT conversation_id
          FROM conversation_topics
          WHERE user_id = OLD.user_id
            AND id = OLD.topic_id
      );
END;
