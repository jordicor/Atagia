-- Phase 1 of memory namespace redesign (see docs/MEMORY_REDESIGN_IMPLEMENTATION_PLAN.md).
-- Additive migration: introduces the new identity, preferences, sensitivity,
-- theme and platform-lock fields side by side with the legacy columns. Old
-- columns and CHECK constraints stay in place until Phase 11 rebuilds the
-- affected tables.
--
-- Tagged-key encoding (rule from the plan):
--     NULL      -> 'n'
--     non-NULL  -> 'v:<length>:<raw value>'
-- Implemented as VIRTUAL generated columns so they can be added with
-- ALTER TABLE and still be referenced from indexes.

-- ---------------------------------------------------------------------------
-- 1. users: cross-chat / cross-device memory preferences.
-- ---------------------------------------------------------------------------
ALTER TABLE users
    ADD COLUMN remember_across_chats INTEGER NOT NULL DEFAULT 1
        CHECK (remember_across_chats IN (0, 1));

ALTER TABLE users
    ADD COLUMN remember_across_devices INTEGER NOT NULL DEFAULT 1
        CHECK (remember_across_devices IN (0, 1));

CREATE INDEX IF NOT EXISTS idx_users_memory_preferences
    ON users(id, remember_across_chats, remember_across_devices);

-- ---------------------------------------------------------------------------
-- 2. conversations: persona / platform / character / mode / incognito.
-- ---------------------------------------------------------------------------
ALTER TABLE conversations ADD COLUMN user_persona_id TEXT;
ALTER TABLE conversations ADD COLUMN platform_id TEXT;
ALTER TABLE conversations ADD COLUMN character_id TEXT;
ALTER TABLE conversations ADD COLUMN mode TEXT;
ALTER TABLE conversations
    ADD COLUMN incognito INTEGER NOT NULL DEFAULT 0
        CHECK (incognito IN (0, 1));

-- Backfill: legacy rows inherit identity from existing columns. The new
-- INTEGER columns have NOT NULL DEFAULT 0 so they are never NULL after the
-- ALTER TABLE; the CASE statements read from the legacy source columns
-- directly instead of treating the new columns as transitional sentinels.
UPDATE conversations
   SET mode = COALESCE(mode, assistant_mode_id),
       character_id = COALESCE(character_id, workspace_id),
       incognito = CASE WHEN isolated_mode = 1 THEN 1 ELSE 0 END,
       platform_id = COALESCE(platform_id, 'default');

CREATE INDEX IF NOT EXISTS idx_conversations_identity
    ON conversations(user_id, user_persona_id, character_id, platform_id, status, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_conversations_incognito_v2
    ON conversations(user_id, incognito, updated_at DESC, id ASC);

-- ---------------------------------------------------------------------------
-- 3. memory_objects: identity, sensitivity, themes, platform lock.
-- ---------------------------------------------------------------------------
ALTER TABLE memory_objects ADD COLUMN user_persona_id TEXT;
ALTER TABLE memory_objects ADD COLUMN platform_id TEXT;
ALTER TABLE memory_objects ADD COLUMN character_id TEXT;

ALTER TABLE memory_objects
    ADD COLUMN sensitivity TEXT NOT NULL DEFAULT 'unknown'
        CHECK (sensitivity IN ('unknown', 'public', 'private', 'secret'));

ALTER TABLE memory_objects
    ADD COLUMN themes_json TEXT NOT NULL DEFAULT '[]';

ALTER TABLE memory_objects
    ADD COLUMN auto_expires INTEGER NOT NULL DEFAULT 0
        CHECK (auto_expires IN (0, 1));

ALTER TABLE memory_objects
    ADD COLUMN platform_locked INTEGER NOT NULL DEFAULT 0
        CHECK (platform_locked IN (0, 1));

ALTER TABLE memory_objects ADD COLUMN platform_id_lock TEXT;

-- Canonical scope column populated alongside the legacy `scope` value.
-- New retrieval filters target `scope_canonical`; Phase 11 collapses both.
ALTER TABLE memory_objects ADD COLUMN scope_canonical TEXT;

-- Backfill existing rows.
UPDATE memory_objects
   SET platform_id = COALESCE(
           platform_id,
           (SELECT c.platform_id FROM conversations c WHERE c.id = memory_objects.conversation_id),
           'default'
       ),
       character_id = COALESCE(
           character_id,
           (SELECT c.character_id FROM conversations c WHERE c.id = memory_objects.conversation_id)
       ),
       scope_canonical = CASE scope
           WHEN 'global_user' THEN 'user'
           WHEN 'conversation' THEN 'chat'
           WHEN 'ephemeral_session' THEN 'chat'
           WHEN 'workspace' THEN 'legacy_workspace'
           WHEN 'assistant_mode' THEN 'legacy_assistant_mode'
           ELSE scope
       END,
       auto_expires = CASE WHEN scope = 'ephemeral_session' THEN 1 ELSE 0 END,
       sensitivity = CASE
           WHEN privacy_level >= 3 THEN 'secret'
           WHEN privacy_level = 2 THEN 'private'
           WHEN privacy_level <= 1 THEN 'public'
           ELSE 'unknown'
       END;

CREATE INDEX IF NOT EXISTS idx_mo_user_persona_scope
    ON memory_objects(user_id, user_persona_id, scope_canonical, status, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_mo_character_scope
    ON memory_objects(user_id, user_persona_id, character_id, scope_canonical, status, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_mo_platform_lock
    ON memory_objects(user_id, platform_locked, platform_id_lock, status);

CREATE INDEX IF NOT EXISTS idx_mo_sensitivity
    ON memory_objects(user_id, sensitivity, status);

-- ---------------------------------------------------------------------------
-- 4. messages: turn-time policy snapshots so delayed workers can apply
--    strictest-wins. Cross-chat raw / source-quote retrieval also needs the
--    sensitivity / theme / platform-lock fields.
-- ---------------------------------------------------------------------------
ALTER TABLE messages ADD COLUMN user_persona_id_snapshot TEXT;
ALTER TABLE messages ADD COLUMN platform_id_snapshot TEXT;
ALTER TABLE messages ADD COLUMN character_id_snapshot TEXT;
ALTER TABLE messages ADD COLUMN mode_snapshot TEXT;
ALTER TABLE messages
    ADD COLUMN incognito_snapshot INTEGER NOT NULL DEFAULT 0
        CHECK (incognito_snapshot IN (0, 1));
ALTER TABLE messages
    ADD COLUMN remember_across_chats_snapshot INTEGER NOT NULL DEFAULT 1
        CHECK (remember_across_chats_snapshot IN (0, 1));
ALTER TABLE messages
    ADD COLUMN remember_across_devices_snapshot INTEGER NOT NULL DEFAULT 1
        CHECK (remember_across_devices_snapshot IN (0, 1));
ALTER TABLE messages
    ADD COLUMN temporary_snapshot INTEGER NOT NULL DEFAULT 0
        CHECK (temporary_snapshot IN (0, 1));
ALTER TABLE messages
    ADD COLUMN purge_on_close_snapshot INTEGER NOT NULL DEFAULT 0
        CHECK (purge_on_close_snapshot IN (0, 1));
ALTER TABLE messages ADD COLUMN valid_to_snapshot TEXT;
ALTER TABLE messages ADD COLUMN policy_snapshot_json TEXT NOT NULL DEFAULT '{}';
ALTER TABLE messages
    ADD COLUMN sensitivity TEXT NOT NULL DEFAULT 'unknown'
        CHECK (sensitivity IN ('unknown', 'public', 'private', 'secret'));
ALTER TABLE messages
    ADD COLUMN themes_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE messages
    ADD COLUMN platform_locked INTEGER NOT NULL DEFAULT 0
        CHECK (platform_locked IN (0, 1));
ALTER TABLE messages ADD COLUMN platform_id_lock TEXT;

-- ---------------------------------------------------------------------------
-- 5. summary_views: identity + sensitivity / theme / platform-lock fields.
-- ---------------------------------------------------------------------------
ALTER TABLE summary_views ADD COLUMN user_persona_id TEXT;
ALTER TABLE summary_views ADD COLUMN platform_id TEXT;
ALTER TABLE summary_views ADD COLUMN character_id TEXT;
ALTER TABLE summary_views
    ADD COLUMN sensitivity TEXT NOT NULL DEFAULT 'unknown'
        CHECK (sensitivity IN ('unknown', 'public', 'private', 'secret'));
ALTER TABLE summary_views
    ADD COLUMN themes_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE summary_views
    ADD COLUMN platform_locked INTEGER NOT NULL DEFAULT 0
        CHECK (platform_locked IN (0, 1));
ALTER TABLE summary_views ADD COLUMN platform_id_lock TEXT;
ALTER TABLE summary_views ADD COLUMN scope_canonical TEXT;

-- ---------------------------------------------------------------------------
-- 6. contract_dimensions_current: identity + sensitivity / themes / platform
--    lock + tagged-key normalized columns for SQLite-safe uniqueness.
-- ---------------------------------------------------------------------------
ALTER TABLE contract_dimensions_current ADD COLUMN user_persona_id TEXT;
ALTER TABLE contract_dimensions_current ADD COLUMN platform_id TEXT;
ALTER TABLE contract_dimensions_current ADD COLUMN character_id TEXT;
ALTER TABLE contract_dimensions_current
    ADD COLUMN sensitivity TEXT NOT NULL DEFAULT 'unknown'
        CHECK (sensitivity IN ('unknown', 'public', 'private', 'secret'));
ALTER TABLE contract_dimensions_current
    ADD COLUMN themes_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE contract_dimensions_current
    ADD COLUMN platform_locked INTEGER NOT NULL DEFAULT 0
        CHECK (platform_locked IN (0, 1));
ALTER TABLE contract_dimensions_current ADD COLUMN platform_id_lock TEXT;
ALTER TABLE contract_dimensions_current ADD COLUMN scope_canonical TEXT;
ALTER TABLE contract_dimensions_current
    ADD COLUMN incognito_snapshot INTEGER NOT NULL DEFAULT 0
        CHECK (incognito_snapshot IN (0, 1));
ALTER TABLE contract_dimensions_current
    ADD COLUMN remember_across_chats_snapshot INTEGER NOT NULL DEFAULT 1
        CHECK (remember_across_chats_snapshot IN (0, 1));
ALTER TABLE contract_dimensions_current
    ADD COLUMN remember_across_devices_snapshot INTEGER NOT NULL DEFAULT 1
        CHECK (remember_across_devices_snapshot IN (0, 1));
ALTER TABLE contract_dimensions_current
    ADD COLUMN temporary_snapshot INTEGER NOT NULL DEFAULT 0
        CHECK (temporary_snapshot IN (0, 1));
ALTER TABLE contract_dimensions_current
    ADD COLUMN purge_on_close_snapshot INTEGER NOT NULL DEFAULT 0
        CHECK (purge_on_close_snapshot IN (0, 1));
ALTER TABLE contract_dimensions_current
    ADD COLUMN policy_snapshot_json TEXT NOT NULL DEFAULT '{}';

ALTER TABLE contract_dimensions_current
    ADD COLUMN user_persona_key TEXT
        GENERATED ALWAYS AS (
            CASE WHEN user_persona_id IS NULL THEN 'n'
                 ELSE 'v:' || length(user_persona_id) || ':' || user_persona_id
            END
        ) VIRTUAL;

ALTER TABLE contract_dimensions_current
    ADD COLUMN character_key TEXT
        GENERATED ALWAYS AS (
            CASE WHEN character_id IS NULL THEN 'n'
                 ELSE 'v:' || length(character_id) || ':' || character_id
            END
        ) VIRTUAL;

ALTER TABLE contract_dimensions_current
    ADD COLUMN conversation_key TEXT
        GENERATED ALWAYS AS (
            CASE WHEN conversation_id IS NULL THEN 'n'
                 ELSE 'v:' || length(conversation_id) || ':' || conversation_id
            END
        ) VIRTUAL;

-- ---------------------------------------------------------------------------
-- 7. consequence_chains: identity + sensitivity / themes / platform lock +
--    source-turn policy snapshots.
-- ---------------------------------------------------------------------------
ALTER TABLE consequence_chains ADD COLUMN user_persona_id TEXT;
ALTER TABLE consequence_chains ADD COLUMN platform_id TEXT;
ALTER TABLE consequence_chains ADD COLUMN character_id TEXT;
ALTER TABLE consequence_chains
    ADD COLUMN sensitivity TEXT NOT NULL DEFAULT 'unknown'
        CHECK (sensitivity IN ('unknown', 'public', 'private', 'secret'));
ALTER TABLE consequence_chains
    ADD COLUMN themes_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE consequence_chains
    ADD COLUMN platform_locked INTEGER NOT NULL DEFAULT 0
        CHECK (platform_locked IN (0, 1));
ALTER TABLE consequence_chains ADD COLUMN platform_id_lock TEXT;
ALTER TABLE consequence_chains ADD COLUMN scope_canonical TEXT;
ALTER TABLE consequence_chains
    ADD COLUMN incognito_snapshot INTEGER NOT NULL DEFAULT 0
        CHECK (incognito_snapshot IN (0, 1));
ALTER TABLE consequence_chains
    ADD COLUMN policy_snapshot_json TEXT NOT NULL DEFAULT '{}';

-- ---------------------------------------------------------------------------
-- 8. graph_entities + graph_relationships + graph side tables.
-- ---------------------------------------------------------------------------
ALTER TABLE graph_entities ADD COLUMN user_persona_id TEXT;
ALTER TABLE graph_entities ADD COLUMN platform_id TEXT;
ALTER TABLE graph_entities ADD COLUMN character_id TEXT;
ALTER TABLE graph_entities
    ADD COLUMN sensitivity TEXT NOT NULL DEFAULT 'unknown'
        CHECK (sensitivity IN ('unknown', 'public', 'private', 'secret'));
ALTER TABLE graph_entities
    ADD COLUMN themes_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE graph_entities
    ADD COLUMN platform_locked INTEGER NOT NULL DEFAULT 0
        CHECK (platform_locked IN (0, 1));
ALTER TABLE graph_entities ADD COLUMN platform_id_lock TEXT;

ALTER TABLE graph_relationships ADD COLUMN user_persona_id TEXT;
ALTER TABLE graph_relationships ADD COLUMN platform_id TEXT;
ALTER TABLE graph_relationships ADD COLUMN character_id TEXT;
ALTER TABLE graph_relationships
    ADD COLUMN sensitivity TEXT NOT NULL DEFAULT 'unknown'
        CHECK (sensitivity IN ('unknown', 'public', 'private', 'secret'));
ALTER TABLE graph_relationships
    ADD COLUMN themes_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE graph_relationships
    ADD COLUMN platform_locked INTEGER NOT NULL DEFAULT 0
        CHECK (platform_locked IN (0, 1));
ALTER TABLE graph_relationships ADD COLUMN platform_id_lock TEXT;
ALTER TABLE graph_relationships ADD COLUMN scope_canonical TEXT;

ALTER TABLE graph_entity_aliases ADD COLUMN user_persona_id TEXT;
ALTER TABLE graph_entity_aliases ADD COLUMN platform_id TEXT;
ALTER TABLE graph_entity_aliases ADD COLUMN character_id TEXT;
ALTER TABLE graph_entity_aliases
    ADD COLUMN sensitivity TEXT NOT NULL DEFAULT 'unknown'
        CHECK (sensitivity IN ('unknown', 'public', 'private', 'secret'));
ALTER TABLE graph_entity_aliases
    ADD COLUMN themes_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE graph_entity_aliases
    ADD COLUMN platform_locked INTEGER NOT NULL DEFAULT 0
        CHECK (platform_locked IN (0, 1));
ALTER TABLE graph_entity_aliases ADD COLUMN platform_id_lock TEXT;

ALTER TABLE graph_entity_mentions ADD COLUMN user_persona_id TEXT;
ALTER TABLE graph_entity_mentions ADD COLUMN platform_id TEXT;
ALTER TABLE graph_entity_mentions ADD COLUMN character_id TEXT;
ALTER TABLE graph_entity_mentions
    ADD COLUMN sensitivity TEXT NOT NULL DEFAULT 'unknown'
        CHECK (sensitivity IN ('unknown', 'public', 'private', 'secret'));
ALTER TABLE graph_entity_mentions
    ADD COLUMN themes_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE graph_entity_mentions
    ADD COLUMN platform_locked INTEGER NOT NULL DEFAULT 0
        CHECK (platform_locked IN (0, 1));
ALTER TABLE graph_entity_mentions ADD COLUMN platform_id_lock TEXT;

ALTER TABLE graph_relationship_sources ADD COLUMN user_persona_id TEXT;
ALTER TABLE graph_relationship_sources ADD COLUMN platform_id TEXT;
ALTER TABLE graph_relationship_sources ADD COLUMN character_id TEXT;
ALTER TABLE graph_relationship_sources
    ADD COLUMN sensitivity TEXT NOT NULL DEFAULT 'unknown'
        CHECK (sensitivity IN ('unknown', 'public', 'private', 'secret'));
ALTER TABLE graph_relationship_sources
    ADD COLUMN themes_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE graph_relationship_sources
    ADD COLUMN platform_locked INTEGER NOT NULL DEFAULT 0
        CHECK (platform_locked IN (0, 1));
ALTER TABLE graph_relationship_sources ADD COLUMN platform_id_lock TEXT;

-- ---------------------------------------------------------------------------
-- 9. memory_links: identity + sensitivity / platform lock + source-turn policy.
-- ---------------------------------------------------------------------------
ALTER TABLE memory_links ADD COLUMN user_persona_id TEXT;
ALTER TABLE memory_links ADD COLUMN platform_id TEXT;
ALTER TABLE memory_links ADD COLUMN character_id TEXT;
ALTER TABLE memory_links ADD COLUMN conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL;
ALTER TABLE memory_links
    ADD COLUMN sensitivity TEXT NOT NULL DEFAULT 'unknown'
        CHECK (sensitivity IN ('unknown', 'public', 'private', 'secret'));
ALTER TABLE memory_links
    ADD COLUMN platform_locked INTEGER NOT NULL DEFAULT 0
        CHECK (platform_locked IN (0, 1));
ALTER TABLE memory_links ADD COLUMN platform_id_lock TEXT;
ALTER TABLE memory_links ADD COLUMN policy_snapshot_json TEXT NOT NULL DEFAULT '{}';

-- ---------------------------------------------------------------------------
-- 10. verbatim_pins: identity + sensitivity / themes / platform lock + tagged keys.
-- ---------------------------------------------------------------------------
ALTER TABLE verbatim_pins ADD COLUMN user_persona_id TEXT;
ALTER TABLE verbatim_pins ADD COLUMN platform_id TEXT;
ALTER TABLE verbatim_pins ADD COLUMN character_id TEXT;
ALTER TABLE verbatim_pins
    ADD COLUMN sensitivity TEXT NOT NULL DEFAULT 'unknown'
        CHECK (sensitivity IN ('unknown', 'public', 'private', 'secret'));
ALTER TABLE verbatim_pins
    ADD COLUMN themes_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE verbatim_pins
    ADD COLUMN platform_locked INTEGER NOT NULL DEFAULT 0
        CHECK (platform_locked IN (0, 1));
ALTER TABLE verbatim_pins ADD COLUMN platform_id_lock TEXT;
ALTER TABLE verbatim_pins ADD COLUMN scope_canonical TEXT;
ALTER TABLE verbatim_pins
    ADD COLUMN incognito_snapshot INTEGER NOT NULL DEFAULT 0
        CHECK (incognito_snapshot IN (0, 1));
ALTER TABLE verbatim_pins
    ADD COLUMN remember_across_chats_snapshot INTEGER NOT NULL DEFAULT 1
        CHECK (remember_across_chats_snapshot IN (0, 1));
ALTER TABLE verbatim_pins
    ADD COLUMN remember_across_devices_snapshot INTEGER NOT NULL DEFAULT 1
        CHECK (remember_across_devices_snapshot IN (0, 1));
ALTER TABLE verbatim_pins
    ADD COLUMN policy_snapshot_json TEXT NOT NULL DEFAULT '{}';

ALTER TABLE verbatim_pins
    ADD COLUMN user_persona_key TEXT
        GENERATED ALWAYS AS (
            CASE WHEN user_persona_id IS NULL THEN 'n'
                 ELSE 'v:' || length(user_persona_id) || ':' || user_persona_id
            END
        ) VIRTUAL;

ALTER TABLE verbatim_pins
    ADD COLUMN character_key TEXT
        GENERATED ALWAYS AS (
            CASE WHEN character_id IS NULL THEN 'n'
                 ELSE 'v:' || length(character_id) || ':' || character_id
            END
        ) VIRTUAL;

ALTER TABLE verbatim_pins
    ADD COLUMN conversation_key TEXT
        GENERATED ALWAYS AS (
            CASE WHEN conversation_id IS NULL THEN 'n'
                 ELSE 'v:' || length(conversation_id) || ':' || conversation_id
            END
        ) VIRTUAL;

-- ---------------------------------------------------------------------------
-- 11. conversation_activity_stats: identity + incognito + effective policy.
-- ---------------------------------------------------------------------------
ALTER TABLE conversation_activity_stats ADD COLUMN user_persona_id TEXT;
ALTER TABLE conversation_activity_stats ADD COLUMN platform_id TEXT;
ALTER TABLE conversation_activity_stats ADD COLUMN character_id TEXT;
ALTER TABLE conversation_activity_stats
    ADD COLUMN incognito INTEGER NOT NULL DEFAULT 0
        CHECK (incognito IN (0, 1));
ALTER TABLE conversation_activity_stats
    ADD COLUMN remember_across_chats INTEGER NOT NULL DEFAULT 1
        CHECK (remember_across_chats IN (0, 1));
ALTER TABLE conversation_activity_stats
    ADD COLUMN remember_across_devices INTEGER NOT NULL DEFAULT 1
        CHECK (remember_across_devices IN (0, 1));
ALTER TABLE conversation_activity_stats
    ADD COLUMN effective_policy_hash TEXT;

CREATE INDEX IF NOT EXISTS idx_conversation_activity_identity
    ON conversation_activity_stats(
        user_id, user_persona_id, character_id, platform_id, incognito,
        likely_soon_score DESC, last_message_at DESC, conversation_id ASC
    );

-- ---------------------------------------------------------------------------
-- 12. retrieval_events: identity + mode + incognito + preferences (trace).
-- ---------------------------------------------------------------------------
ALTER TABLE retrieval_events ADD COLUMN user_persona_id TEXT;
ALTER TABLE retrieval_events ADD COLUMN platform_id TEXT;
ALTER TABLE retrieval_events ADD COLUMN character_id TEXT;
ALTER TABLE retrieval_events ADD COLUMN mode TEXT;
ALTER TABLE retrieval_events
    ADD COLUMN incognito INTEGER NOT NULL DEFAULT 0
        CHECK (incognito IN (0, 1));
ALTER TABLE retrieval_events
    ADD COLUMN remember_across_chats INTEGER NOT NULL DEFAULT 1
        CHECK (remember_across_chats IN (0, 1));
ALTER TABLE retrieval_events
    ADD COLUMN remember_across_devices INTEGER NOT NULL DEFAULT 1
        CHECK (remember_across_devices IN (0, 1));

-- ---------------------------------------------------------------------------
-- 13. memory_feedback_events: identity + mode + snapshots.
-- ---------------------------------------------------------------------------
ALTER TABLE memory_feedback_events ADD COLUMN user_persona_id TEXT;
ALTER TABLE memory_feedback_events ADD COLUMN platform_id TEXT;
ALTER TABLE memory_feedback_events ADD COLUMN character_id TEXT;
ALTER TABLE memory_feedback_events ADD COLUMN conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL;
ALTER TABLE memory_feedback_events ADD COLUMN mode TEXT;
ALTER TABLE memory_feedback_events
    ADD COLUMN incognito_snapshot INTEGER NOT NULL DEFAULT 0
        CHECK (incognito_snapshot IN (0, 1));
ALTER TABLE memory_feedback_events
    ADD COLUMN remember_across_chats_snapshot INTEGER NOT NULL DEFAULT 1
        CHECK (remember_across_chats_snapshot IN (0, 1));
ALTER TABLE memory_feedback_events
    ADD COLUMN remember_across_devices_snapshot INTEGER NOT NULL DEFAULT 1
        CHECK (remember_across_devices_snapshot IN (0, 1));
ALTER TABLE memory_feedback_events
    ADD COLUMN policy_snapshot_json TEXT NOT NULL DEFAULT '{}';

-- ---------------------------------------------------------------------------
-- 14. conversation_topics: identity + sensitivity / themes / platform lock.
-- ---------------------------------------------------------------------------
ALTER TABLE conversation_topics ADD COLUMN user_persona_id TEXT;
ALTER TABLE conversation_topics ADD COLUMN platform_id TEXT;
ALTER TABLE conversation_topics ADD COLUMN character_id TEXT;
ALTER TABLE conversation_topics
    ADD COLUMN sensitivity TEXT NOT NULL DEFAULT 'unknown'
        CHECK (sensitivity IN ('unknown', 'public', 'private', 'secret'));
ALTER TABLE conversation_topics
    ADD COLUMN themes_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE conversation_topics
    ADD COLUMN platform_locked INTEGER NOT NULL DEFAULT 0
        CHECK (platform_locked IN (0, 1));
ALTER TABLE conversation_topics ADD COLUMN platform_id_lock TEXT;

-- ---------------------------------------------------------------------------
-- 15. memory_consent_profile: persona-scoped consent + tagged key.
-- ---------------------------------------------------------------------------
ALTER TABLE memory_consent_profile ADD COLUMN user_persona_id TEXT;

ALTER TABLE memory_consent_profile
    ADD COLUMN user_persona_key TEXT
        GENERATED ALWAYS AS (
            CASE WHEN user_persona_id IS NULL THEN 'n'
                 ELSE 'v:' || length(user_persona_id) || ':' || user_persona_id
            END
        ) VIRTUAL;

-- The original memory_consent_profile schema declares
-- `UNIQUE(user_id, category)` as an inline column constraint, which SQLite
-- materializes as an autoindex that cannot be dropped via DROP INDEX. The
-- table rebuild that removes it lives in Phase 11 (migration 0032+); this
-- migration only adds the new persona column and key.
CREATE INDEX IF NOT EXISTS idx_consent_profile_persona_key
    ON memory_consent_profile(user_id, user_persona_key, category);

-- ---------------------------------------------------------------------------
-- 16. pending_memory_confirmations: full source-turn snapshots.
-- ---------------------------------------------------------------------------
ALTER TABLE pending_memory_confirmations ADD COLUMN user_persona_id TEXT;
ALTER TABLE pending_memory_confirmations ADD COLUMN platform_id TEXT;
ALTER TABLE pending_memory_confirmations ADD COLUMN character_id TEXT;
ALTER TABLE pending_memory_confirmations ADD COLUMN mode TEXT;
ALTER TABLE pending_memory_confirmations
    ADD COLUMN incognito_snapshot INTEGER NOT NULL DEFAULT 0
        CHECK (incognito_snapshot IN (0, 1));
ALTER TABLE pending_memory_confirmations
    ADD COLUMN remember_across_chats_snapshot INTEGER NOT NULL DEFAULT 1
        CHECK (remember_across_chats_snapshot IN (0, 1));
ALTER TABLE pending_memory_confirmations
    ADD COLUMN remember_across_devices_snapshot INTEGER NOT NULL DEFAULT 1
        CHECK (remember_across_devices_snapshot IN (0, 1));
ALTER TABLE pending_memory_confirmations
    ADD COLUMN temporary_snapshot INTEGER NOT NULL DEFAULT 0
        CHECK (temporary_snapshot IN (0, 1));
ALTER TABLE pending_memory_confirmations
    ADD COLUMN purge_on_close_snapshot INTEGER NOT NULL DEFAULT 0
        CHECK (purge_on_close_snapshot IN (0, 1));
ALTER TABLE pending_memory_confirmations ADD COLUMN valid_to_snapshot TEXT;
ALTER TABLE pending_memory_confirmations ADD COLUMN intended_scope TEXT;
ALTER TABLE pending_memory_confirmations
    ADD COLUMN intended_sensitivity TEXT NOT NULL DEFAULT 'unknown'
        CHECK (intended_sensitivity IN ('unknown', 'public', 'private', 'secret'));
ALTER TABLE pending_memory_confirmations
    ADD COLUMN platform_locked INTEGER NOT NULL DEFAULT 0
        CHECK (platform_locked IN (0, 1));
ALTER TABLE pending_memory_confirmations ADD COLUMN platform_id_lock TEXT;
ALTER TABLE pending_memory_confirmations ADD COLUMN policy_snapshot_json TEXT NOT NULL DEFAULT '{}';
-- A pending row is non-activatable until source-turn policy can be proven.
ALTER TABLE pending_memory_confirmations
    ADD COLUMN policy_proven INTEGER NOT NULL DEFAULT 0
        CHECK (policy_proven IN (0, 1));

-- ---------------------------------------------------------------------------
-- 17. worker_job_runs: identity + snapshots.
-- ---------------------------------------------------------------------------
ALTER TABLE worker_job_runs ADD COLUMN user_persona_id TEXT;
ALTER TABLE worker_job_runs ADD COLUMN platform_id TEXT;
ALTER TABLE worker_job_runs ADD COLUMN character_id TEXT;
ALTER TABLE worker_job_runs
    ADD COLUMN incognito_snapshot INTEGER NOT NULL DEFAULT 0
        CHECK (incognito_snapshot IN (0, 1));
ALTER TABLE worker_job_runs
    ADD COLUMN remember_across_chats_snapshot INTEGER NOT NULL DEFAULT 1
        CHECK (remember_across_chats_snapshot IN (0, 1));
ALTER TABLE worker_job_runs
    ADD COLUMN remember_across_devices_snapshot INTEGER NOT NULL DEFAULT 1
        CHECK (remember_across_devices_snapshot IN (0, 1));
ALTER TABLE worker_job_runs
    ADD COLUMN temporary_snapshot INTEGER NOT NULL DEFAULT 0
        CHECK (temporary_snapshot IN (0, 1));
ALTER TABLE worker_job_runs
    ADD COLUMN purge_on_close_snapshot INTEGER NOT NULL DEFAULT 0
        CHECK (purge_on_close_snapshot IN (0, 1));
ALTER TABLE worker_job_runs ADD COLUMN policy_snapshot_json TEXT NOT NULL DEFAULT '{}';

-- ---------------------------------------------------------------------------
-- 18. memory_embedding_metadata: namespace metadata for vector partitioning.
-- ---------------------------------------------------------------------------
ALTER TABLE memory_embedding_metadata ADD COLUMN user_persona_id TEXT;
ALTER TABLE memory_embedding_metadata ADD COLUMN platform_id TEXT;
ALTER TABLE memory_embedding_metadata ADD COLUMN character_id TEXT;
ALTER TABLE memory_embedding_metadata
    ADD COLUMN sensitivity TEXT NOT NULL DEFAULT 'unknown'
        CHECK (sensitivity IN ('unknown', 'public', 'private', 'secret'));
ALTER TABLE memory_embedding_metadata
    ADD COLUMN platform_locked INTEGER NOT NULL DEFAULT 0
        CHECK (platform_locked IN (0, 1));
ALTER TABLE memory_embedding_metadata ADD COLUMN platform_id_lock TEXT;
ALTER TABLE memory_embedding_metadata ADD COLUMN scope_canonical TEXT;

CREATE INDEX IF NOT EXISTS idx_embedding_metadata_persona
    ON memory_embedding_metadata(user_id, user_persona_id, character_id, platform_id, sensitivity);
