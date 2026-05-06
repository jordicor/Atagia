-- atagia:foreign_keys_off

-- Phase 11: final data-model cut for canonical memory scopes.

PRAGMA legacy_alter_table = ON;

CREATE TABLE IF NOT EXISTS memory_redesign_phase11_report (
    table_name TEXT NOT NULL,
    disposition TEXT NOT NULL,
    affected_count INTEGER NOT NULL,
    sample_ids_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);

CREATE TEMP TABLE phase11_dropped_memory_ids AS
SELECT id
FROM memory_objects
WHERE COALESCE(
        scope_canonical,
        CASE
            WHEN scope = 'global_user' THEN 'user'
            WHEN scope = 'assistant_mode' THEN 'legacy_assistant_mode'
            WHEN scope = 'workspace' THEN 'legacy_workspace'
            WHEN scope IN ('conversation', 'ephemeral_session') THEN 'chat'
            ELSE scope
        END
    ) IN ('legacy_assistant_mode', 'legacy_workspace');

INSERT INTO memory_redesign_phase11_report(table_name, disposition, affected_count, sample_ids_json, created_at)
SELECT
    'memory_objects',
    'dropped_legacy_scope_rows',
    COUNT(*),
    COALESCE((SELECT json_group_array(id) FROM (SELECT id FROM phase11_dropped_memory_ids ORDER BY id LIMIT 20)), '[]'),
    datetime('now')
FROM phase11_dropped_memory_ids;

DELETE FROM belief_versions
WHERE belief_id IN (SELECT id FROM phase11_dropped_memory_ids);

DELETE FROM memory_feedback_events
WHERE memory_id IN (SELECT id FROM phase11_dropped_memory_ids);

DELETE FROM memory_edit_history
WHERE memory_id IN (SELECT id FROM phase11_dropped_memory_ids);

DELETE FROM pending_memory_confirmations
WHERE memory_id IN (SELECT id FROM phase11_dropped_memory_ids);

DELETE FROM memory_embedding_metadata
WHERE memory_id IN (SELECT id FROM phase11_dropped_memory_ids);

DELETE FROM graph_entity_mentions
WHERE memory_id IN (SELECT id FROM phase11_dropped_memory_ids);

DELETE FROM graph_relationship_sources
WHERE memory_id IN (SELECT id FROM phase11_dropped_memory_ids);

DELETE FROM consequence_chains
WHERE action_memory_id IN (SELECT id FROM phase11_dropped_memory_ids)
   OR outcome_memory_id IN (SELECT id FROM phase11_dropped_memory_ids)
   OR tendency_belief_id IN (SELECT id FROM phase11_dropped_memory_ids);

DROP TRIGGER IF EXISTS memory_objects_fts_ai;
DROP TRIGGER IF EXISTS memory_objects_fts_ad;
DROP TRIGGER IF EXISTS memory_objects_fts_au;
DROP TABLE IF EXISTS memory_objects_fts;

ALTER TABLE memory_objects RENAME TO memory_objects_old;

CREATE TABLE memory_objects (
    _rowid INTEGER PRIMARY KEY,
    id TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    workspace_id TEXT REFERENCES workspaces(id) ON DELETE SET NULL,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
    assistant_mode_id TEXT REFERENCES assistant_modes(id) ON DELETE SET NULL,
    object_type TEXT NOT NULL CHECK (
        object_type IN ('evidence', 'belief', 'interaction_contract', 'state_snapshot', 'consequence_chain', 'summary_view')
    ),
    scope TEXT NOT NULL CHECK (scope IN ('chat', 'character', 'user')),
    canonical_text TEXT NOT NULL,
    index_text TEXT,
    extraction_hash TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}',
    source_kind TEXT NOT NULL CHECK (source_kind IN ('verbatim', 'extracted', 'inferred', 'summarized', 'composed')),
    confidence REAL NOT NULL DEFAULT 0.5,
    stability REAL NOT NULL DEFAULT 0.5,
    vitality REAL NOT NULL DEFAULT 0.0,
    maya_score REAL NOT NULL DEFAULT 0.0,
    privacy_level INTEGER NOT NULL DEFAULT 0,
    memory_category TEXT NOT NULL DEFAULT 'unknown' CHECK (
        memory_category IN ('phone', 'address', 'pin_or_password', 'medication', 'financial', 'date_of_birth', 'contact_identity', 'other_sensitive', 'unknown')
    ),
    preserve_verbatim INTEGER NOT NULL DEFAULT 0 CHECK (preserve_verbatim IN (0, 1)),
    valid_from TEXT,
    valid_to TEXT,
    temporal_type TEXT NOT NULL DEFAULT 'unknown',
    tension_score REAL NOT NULL DEFAULT 0.0,
    tension_updated_at TEXT,
    status TEXT NOT NULL DEFAULT 'active' CHECK (
        status IN ('active', 'superseded', 'archived', 'deleted', 'review_required', 'pending_user_confirmation', 'declined')
    ),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    language_codes_json TEXT,
    intimacy_boundary TEXT NOT NULL DEFAULT 'ordinary' CHECK (
        intimacy_boundary IN ('ordinary', 'romantic_private', 'intimacy_private', 'intimacy_preference_private', 'intimacy_boundary', 'ambiguous_intimate', 'safety_blocked')
    ),
    intimacy_boundary_confidence REAL NOT NULL DEFAULT 0.0 CHECK (intimacy_boundary_confidence BETWEEN 0.0 AND 1.0),
    archived_by_conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
    user_persona_id TEXT,
    platform_id TEXT,
    character_id TEXT,
    sensitivity TEXT NOT NULL DEFAULT 'unknown' CHECK (sensitivity IN ('unknown', 'public', 'private', 'secret')),
    themes_json TEXT NOT NULL DEFAULT '[]',
    auto_expires INTEGER NOT NULL DEFAULT 0 CHECK (auto_expires IN (0, 1)),
    platform_locked INTEGER NOT NULL DEFAULT 0 CHECK (platform_locked IN (0, 1)),
    platform_id_lock TEXT,
    scope_canonical TEXT
);

INSERT INTO memory_objects(
    _rowid, id, user_id, workspace_id, conversation_id, assistant_mode_id,
    object_type, scope, canonical_text, index_text, extraction_hash, payload_json,
    source_kind, confidence, stability, vitality, maya_score, privacy_level,
    memory_category, preserve_verbatim, valid_from, valid_to, temporal_type,
    tension_score, tension_updated_at, status, created_at, updated_at,
    language_codes_json, intimacy_boundary, intimacy_boundary_confidence,
    archived_by_conversation_id, user_persona_id, platform_id, character_id,
    sensitivity, themes_json, auto_expires, platform_locked, platform_id_lock,
    scope_canonical
)
SELECT
    _rowid,
    id,
    user_id,
    workspace_id,
    conversation_id,
    assistant_mode_id,
    object_type,
    CASE canonical_scope WHEN 'chat' THEN 'chat' WHEN 'character' THEN 'character' ELSE 'user' END,
    canonical_text,
    index_text,
    extraction_hash,
    payload_json,
    source_kind,
    confidence,
    stability,
    vitality,
    maya_score,
    privacy_level,
    memory_category,
    preserve_verbatim,
    valid_from,
    valid_to,
    temporal_type,
    tension_score,
    tension_updated_at,
    status,
    created_at,
    updated_at,
    language_codes_json,
    intimacy_boundary,
    intimacy_boundary_confidence,
    archived_by_conversation_id,
    user_persona_id,
    platform_id,
    character_id,
    sensitivity,
    themes_json,
    auto_expires,
    platform_locked,
    platform_id_lock,
    CASE canonical_scope WHEN 'chat' THEN 'chat' WHEN 'character' THEN 'character' ELSE 'user' END
FROM (
    SELECT
        *,
        COALESCE(
            scope_canonical,
            CASE
                WHEN scope = 'global_user' THEN 'user'
                WHEN scope = 'assistant_mode' THEN 'legacy_assistant_mode'
                WHEN scope = 'workspace' THEN 'legacy_workspace'
                WHEN scope IN ('conversation', 'ephemeral_session') THEN 'chat'
                ELSE scope
            END
        ) AS canonical_scope
    FROM memory_objects_old
)
WHERE canonical_scope IN ('chat', 'character', 'user');

DROP TABLE memory_objects_old;

CREATE INDEX idx_memory_objects_user_type_scope
    ON memory_objects(user_id, object_type, scope);
CREATE INDEX idx_memory_objects_status
    ON memory_objects(status);
CREATE INDEX idx_memory_objects_user_status
    ON memory_objects(user_id, status);
CREATE UNIQUE INDEX uq_memory_objects_user_extraction_hash
    ON memory_objects(user_id, extraction_hash)
    WHERE extraction_hash IS NOT NULL;
CREATE INDEX idx_mo_tension
    ON memory_objects(user_id, tension_score)
    WHERE tension_score > 0 AND object_type = 'belief';
CREATE INDEX idx_mo_temporal_expiry
    ON memory_objects(valid_to, status, object_type)
    WHERE valid_to IS NOT NULL AND status = 'active';
CREATE INDEX idx_mo_user_status_intimacy
    ON memory_objects(user_id, status, intimacy_boundary);
CREATE INDEX idx_mo_archived_by_conversation
    ON memory_objects(user_id, archived_by_conversation_id, status);
CREATE INDEX idx_mo_user_persona_scope
    ON memory_objects(user_id, user_persona_id, scope_canonical, status, updated_at DESC);
CREATE INDEX idx_mo_character_scope
    ON memory_objects(user_id, user_persona_id, character_id, scope_canonical, status, updated_at DESC);
CREATE INDEX idx_mo_platform_lock
    ON memory_objects(user_id, platform_locked, platform_id_lock, status);
CREATE INDEX idx_mo_sensitivity
    ON memory_objects(user_id, sensitivity, status);

CREATE VIRTUAL TABLE memory_objects_fts
USING fts5(
    canonical_text,
    index_text,
    content='memory_objects',
    content_rowid='_rowid',
    tokenize='unicode61'
);

CREATE TRIGGER memory_objects_fts_ai
AFTER INSERT ON memory_objects
BEGIN
    INSERT INTO memory_objects_fts(rowid, canonical_text, index_text)
    VALUES (new._rowid, new.canonical_text, new.index_text);
END;

CREATE TRIGGER memory_objects_fts_ad
AFTER DELETE ON memory_objects
BEGIN
    INSERT INTO memory_objects_fts(memory_objects_fts, rowid, canonical_text, index_text)
    VALUES ('delete', old._rowid, old.canonical_text, old.index_text);
END;

CREATE TRIGGER memory_objects_fts_au
AFTER UPDATE ON memory_objects
BEGIN
    INSERT INTO memory_objects_fts(memory_objects_fts, rowid, canonical_text, index_text)
    VALUES ('delete', old._rowid, old.canonical_text, old.index_text);
    INSERT INTO memory_objects_fts(rowid, canonical_text, index_text)
    VALUES (new._rowid, new.canonical_text, new.index_text);
END;

INSERT INTO memory_objects_fts(memory_objects_fts) VALUES('rebuild');

DELETE FROM memory_embedding_metadata
WHERE memory_id NOT IN (SELECT id FROM memory_objects);

UPDATE memory_embedding_metadata
SET scope = CASE
        WHEN COALESCE(scope_canonical, scope) IN ('conversation', 'ephemeral_session', 'chat') THEN 'chat'
        WHEN COALESCE(scope_canonical, scope) IN ('workspace', 'character') THEN 'character'
        WHEN COALESCE(scope_canonical, scope) IN ('global_user', 'assistant_mode', 'user') THEN 'user'
        ELSE scope
    END,
    scope_canonical = CASE
        WHEN COALESCE(scope_canonical, scope) IN ('conversation', 'ephemeral_session', 'chat') THEN 'chat'
        WHEN COALESCE(scope_canonical, scope) IN ('workspace', 'character') THEN 'character'
        WHEN COALESCE(scope_canonical, scope) IN ('global_user', 'assistant_mode', 'user') THEN 'user'
        ELSE scope
    END;

DROP TRIGGER IF EXISTS verbatim_pins_fts_ai;
DROP TRIGGER IF EXISTS verbatim_pins_fts_ad;
DROP TRIGGER IF EXISTS verbatim_pins_fts_au;
DROP TABLE IF EXISTS verbatim_pins_fts;

ALTER TABLE verbatim_pins RENAME TO verbatim_pins_old;

CREATE TABLE verbatim_pins (
    _rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    workspace_id TEXT REFERENCES workspaces(id) ON DELETE SET NULL,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
    assistant_mode_id TEXT REFERENCES assistant_modes(id) ON DELETE SET NULL,
    scope TEXT NOT NULL CHECK (scope IN ('chat', 'character', 'user')),
    target_kind TEXT NOT NULL,
    target_id TEXT NOT NULL,
    target_span_start INTEGER,
    target_span_end INTEGER,
    canonical_text TEXT NOT NULL,
    index_text TEXT NOT NULL,
    privacy_level INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'active',
    reason TEXT,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    expires_at TEXT,
    deleted_at TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}',
    intimacy_boundary TEXT NOT NULL DEFAULT 'ordinary' CHECK (
        intimacy_boundary IN ('ordinary', 'romantic_private', 'intimacy_private', 'intimacy_preference_private', 'intimacy_boundary', 'ambiguous_intimate', 'safety_blocked')
    ),
    intimacy_boundary_confidence REAL NOT NULL DEFAULT 0.0 CHECK (intimacy_boundary_confidence BETWEEN 0.0 AND 1.0),
    user_persona_id TEXT,
    platform_id TEXT,
    character_id TEXT,
    sensitivity TEXT NOT NULL DEFAULT 'unknown' CHECK (sensitivity IN ('unknown', 'public', 'private', 'secret')),
    themes_json TEXT NOT NULL DEFAULT '[]',
    platform_locked INTEGER NOT NULL DEFAULT 0 CHECK (platform_locked IN (0, 1)),
    platform_id_lock TEXT,
    scope_canonical TEXT,
    incognito_snapshot INTEGER NOT NULL DEFAULT 0 CHECK (incognito_snapshot IN (0, 1)),
    remember_across_chats_snapshot INTEGER NOT NULL DEFAULT 1 CHECK (remember_across_chats_snapshot IN (0, 1)),
    remember_across_devices_snapshot INTEGER NOT NULL DEFAULT 1 CHECK (remember_across_devices_snapshot IN (0, 1)),
    policy_snapshot_json TEXT NOT NULL DEFAULT '{}',
    user_persona_key TEXT
        GENERATED ALWAYS AS (
            CASE WHEN user_persona_id IS NULL THEN 'n'
                 ELSE 'v:' || length(user_persona_id) || ':' || user_persona_id
            END
        ) VIRTUAL,
    character_key TEXT
        GENERATED ALWAYS AS (
            CASE WHEN character_id IS NULL THEN 'n'
                 ELSE 'v:' || length(character_id) || ':' || character_id
            END
        ) VIRTUAL,
    conversation_key TEXT
        GENERATED ALWAYS AS (
            CASE WHEN conversation_id IS NULL THEN 'n'
                 ELSE 'v:' || length(conversation_id) || ':' || conversation_id
            END
        ) VIRTUAL,
    CHECK (target_kind IN ('message', 'memory_object', 'text_span')),
    CHECK (status IN ('active', 'archived', 'expired', 'deleted')),
    CHECK (privacy_level BETWEEN 0 AND 3),
    CHECK (target_span_start IS NULL OR target_span_start >= 0),
    CHECK (target_span_end IS NULL OR target_span_end >= 0),
    CHECK (target_span_start IS NULL OR target_span_end IS NULL OR target_span_end >= target_span_start)
);

INSERT INTO memory_redesign_phase11_report(table_name, disposition, affected_count, sample_ids_json, created_at)
SELECT
    'verbatim_pins',
    'dropped_legacy_scope_rows',
    COUNT(*),
    COALESCE((SELECT json_group_array(id) FROM (
        SELECT id
        FROM verbatim_pins_old
        WHERE COALESCE(scope_canonical, scope) IN ('workspace', 'legacy_workspace', 'assistant_mode', 'legacy_assistant_mode')
        ORDER BY id
        LIMIT 20
    )), '[]'),
    datetime('now')
FROM verbatim_pins_old
WHERE COALESCE(scope_canonical, scope) IN ('workspace', 'legacy_workspace', 'assistant_mode', 'legacy_assistant_mode');

INSERT INTO verbatim_pins(
    _rowid, id, user_id, workspace_id, conversation_id, assistant_mode_id,
    scope, target_kind, target_id, target_span_start, target_span_end,
    canonical_text, index_text, privacy_level, status, reason, created_by,
    created_at, updated_at, expires_at, deleted_at, payload_json,
    intimacy_boundary, intimacy_boundary_confidence, user_persona_id, platform_id,
    character_id, sensitivity, themes_json, platform_locked, platform_id_lock,
    scope_canonical, incognito_snapshot, remember_across_chats_snapshot,
    remember_across_devices_snapshot, policy_snapshot_json
)
SELECT
    _rowid,
    id,
    user_id,
    workspace_id,
    conversation_id,
    assistant_mode_id,
    canonical_scope,
    target_kind,
    target_id,
    target_span_start,
    target_span_end,
    canonical_text,
    index_text,
    privacy_level,
    status,
    reason,
    created_by,
    created_at,
    updated_at,
    expires_at,
    deleted_at,
    payload_json,
    intimacy_boundary,
    intimacy_boundary_confidence,
    user_persona_id,
    platform_id,
    character_id,
    sensitivity,
    themes_json,
    platform_locked,
    platform_id_lock,
    canonical_scope,
    incognito_snapshot,
    remember_across_chats_snapshot,
    remember_across_devices_snapshot,
    policy_snapshot_json
FROM (
    SELECT
        *,
        CASE
            WHEN COALESCE(scope_canonical, scope) IN ('conversation', 'ephemeral_session', 'chat') THEN 'chat'
            WHEN COALESCE(scope_canonical, scope) = 'character' THEN 'character'
            WHEN COALESCE(scope_canonical, scope) IN ('global_user', 'user') THEN 'user'
            WHEN COALESCE(scope_canonical, scope) IN ('workspace', 'legacy_workspace') THEN 'legacy_workspace'
            WHEN COALESCE(scope_canonical, scope) IN ('assistant_mode', 'legacy_assistant_mode') THEN 'legacy_assistant_mode'
            ELSE 'legacy_unknown'
        END AS canonical_scope
    FROM verbatim_pins_old
)
WHERE canonical_scope IN ('chat', 'character', 'user');

DROP TABLE verbatim_pins_old;

CREATE INDEX verbatim_pins_user_status_scope_target_idx
    ON verbatim_pins(user_id, status, scope, assistant_mode_id, workspace_id, conversation_id, target_kind, target_id);
CREATE INDEX verbatim_pins_user_status_updated_idx
    ON verbatim_pins(user_id, status, updated_at DESC, created_at DESC, id ASC);
CREATE INDEX verbatim_pins_user_expires_idx
    ON verbatim_pins(user_id, expires_at);
CREATE INDEX verbatim_pins_user_status_intimacy_idx
    ON verbatim_pins(user_id, status, intimacy_boundary, updated_at DESC, id ASC);

CREATE VIRTUAL TABLE verbatim_pins_fts
USING fts5(
    index_text,
    content='verbatim_pins',
    content_rowid='_rowid',
    tokenize='unicode61'
);

CREATE TRIGGER verbatim_pins_fts_ai
AFTER INSERT ON verbatim_pins
BEGIN
    INSERT INTO verbatim_pins_fts(rowid, index_text)
    VALUES (new._rowid, new.index_text);
END;

CREATE TRIGGER verbatim_pins_fts_ad
AFTER DELETE ON verbatim_pins
BEGIN
    INSERT INTO verbatim_pins_fts(verbatim_pins_fts, rowid, index_text)
    VALUES ('delete', old._rowid, old.index_text);
END;

CREATE TRIGGER verbatim_pins_fts_au
AFTER UPDATE ON verbatim_pins
BEGIN
    INSERT INTO verbatim_pins_fts(verbatim_pins_fts, rowid, index_text)
    VALUES ('delete', old._rowid, old.index_text);
    INSERT INTO verbatim_pins_fts(rowid, index_text)
    VALUES (new._rowid, new.index_text);
END;

INSERT INTO verbatim_pins_fts(verbatim_pins_fts) VALUES('rebuild');

ALTER TABLE contract_dimensions_current RENAME TO contract_dimensions_current_old;

CREATE TABLE contract_dimensions_current (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    workspace_id TEXT REFERENCES workspaces(id) ON DELETE SET NULL,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
    assistant_mode_id TEXT REFERENCES assistant_modes(id) ON DELETE SET NULL,
    scope TEXT NOT NULL CHECK (scope IN ('chat', 'character', 'user')),
    dimension_name TEXT NOT NULL,
    value_json TEXT NOT NULL,
    confidence REAL NOT NULL,
    source_memory_id TEXT NOT NULL REFERENCES memory_objects(id) ON DELETE CASCADE,
    updated_at TEXT NOT NULL,
    user_persona_id TEXT,
    platform_id TEXT,
    character_id TEXT,
    sensitivity TEXT NOT NULL DEFAULT 'unknown' CHECK (sensitivity IN ('unknown', 'public', 'private', 'secret')),
    themes_json TEXT NOT NULL DEFAULT '[]',
    platform_locked INTEGER NOT NULL DEFAULT 0 CHECK (platform_locked IN (0, 1)),
    platform_id_lock TEXT,
    scope_canonical TEXT,
    incognito_snapshot INTEGER NOT NULL DEFAULT 0 CHECK (incognito_snapshot IN (0, 1)),
    remember_across_chats_snapshot INTEGER NOT NULL DEFAULT 1 CHECK (remember_across_chats_snapshot IN (0, 1)),
    remember_across_devices_snapshot INTEGER NOT NULL DEFAULT 1 CHECK (remember_across_devices_snapshot IN (0, 1)),
    temporary_snapshot INTEGER NOT NULL DEFAULT 0 CHECK (temporary_snapshot IN (0, 1)),
    purge_on_close_snapshot INTEGER NOT NULL DEFAULT 0 CHECK (purge_on_close_snapshot IN (0, 1)),
    policy_snapshot_json TEXT NOT NULL DEFAULT '{}',
    user_persona_key TEXT
        GENERATED ALWAYS AS (
            CASE WHEN user_persona_id IS NULL THEN 'n'
                 ELSE 'v:' || length(user_persona_id) || ':' || user_persona_id
            END
        ) VIRTUAL,
    character_key TEXT
        GENERATED ALWAYS AS (
            CASE WHEN character_id IS NULL THEN 'n'
                 ELSE 'v:' || length(character_id) || ':' || character_id
            END
        ) VIRTUAL,
    conversation_key TEXT
        GENERATED ALWAYS AS (
            CASE WHEN conversation_id IS NULL THEN 'n'
                 ELSE 'v:' || length(conversation_id) || ':' || conversation_id
            END
        ) VIRTUAL,
    scope_canonical_key TEXT
        GENERATED ALWAYS AS (
            CASE WHEN scope_canonical IS NULL THEN scope ELSE scope_canonical END
        ) VIRTUAL
);

INSERT INTO contract_dimensions_current(
    id, user_id, workspace_id, conversation_id, assistant_mode_id, scope,
    dimension_name, value_json, confidence, source_memory_id, updated_at,
    user_persona_id, platform_id, character_id, sensitivity, themes_json,
    platform_locked, platform_id_lock, scope_canonical, incognito_snapshot,
    remember_across_chats_snapshot, remember_across_devices_snapshot,
    temporary_snapshot, purge_on_close_snapshot, policy_snapshot_json
)
SELECT
    id, user_id, workspace_id, conversation_id, assistant_mode_id, canonical_scope,
    dimension_name, value_json, confidence, source_memory_id, updated_at,
    user_persona_id, platform_id, character_id, sensitivity, themes_json,
    platform_locked, platform_id_lock, canonical_scope, incognito_snapshot,
    remember_across_chats_snapshot, remember_across_devices_snapshot,
    temporary_snapshot, purge_on_close_snapshot, policy_snapshot_json
FROM (
    SELECT
        cdc.*,
        CASE
            WHEN COALESCE(scope_canonical, scope) IN ('conversation', 'ephemeral_session', 'chat') THEN 'chat'
            WHEN COALESCE(scope_canonical, scope) IN ('workspace', 'legacy_workspace') THEN 'legacy_workspace'
            WHEN COALESCE(scope_canonical, scope) IN ('assistant_mode', 'legacy_assistant_mode') THEN 'legacy_assistant_mode'
            WHEN COALESCE(scope_canonical, scope) IN ('character') THEN 'character'
            ELSE 'user'
        END AS canonical_scope,
        ROW_NUMBER() OVER (
            PARTITION BY
                user_id,
                CASE WHEN user_persona_id IS NULL THEN 'n' ELSE 'v:' || length(user_persona_id) || ':' || user_persona_id END,
                CASE WHEN character_id IS NULL THEN 'n' ELSE 'v:' || length(character_id) || ':' || character_id END,
                CASE WHEN conversation_id IS NULL THEN 'n' ELSE 'v:' || length(conversation_id) || ':' || conversation_id END,
                CASE
                    WHEN COALESCE(scope_canonical, scope) IN ('conversation', 'ephemeral_session', 'chat') THEN 'chat'
                    WHEN COALESCE(scope_canonical, scope) IN ('workspace', 'legacy_workspace') THEN 'legacy_workspace'
                    WHEN COALESCE(scope_canonical, scope) IN ('assistant_mode', 'legacy_assistant_mode') THEN 'legacy_assistant_mode'
                    WHEN COALESCE(scope_canonical, scope) IN ('character') THEN 'character'
                    ELSE 'user'
                END,
                dimension_name
            ORDER BY confidence DESC, updated_at DESC, id ASC
        ) AS rn
    FROM contract_dimensions_current_old AS cdc
    WHERE source_memory_id IN (SELECT id FROM memory_objects)
)
WHERE rn = 1
  AND canonical_scope IN ('chat', 'character', 'user');

DROP TABLE contract_dimensions_current_old;

CREATE INDEX idx_contract_dims_user
    ON contract_dimensions_current(user_id);
CREATE UNIQUE INDEX uq_contract_dimensions_current
    ON contract_dimensions_current(
        user_id,
        user_persona_key,
        character_key,
        conversation_key,
        scope_canonical_key,
        dimension_name
    );

ALTER TABLE memory_consent_profile RENAME TO memory_consent_profile_old;

CREATE TABLE memory_consent_profile (
    _rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    category TEXT NOT NULL CHECK (
        category IN ('phone', 'address', 'pin_or_password', 'medication', 'financial', 'date_of_birth', 'contact_identity', 'other_sensitive', 'unknown')
    ),
    confirmed_count INTEGER NOT NULL DEFAULT 0,
    declined_count INTEGER NOT NULL DEFAULT 0,
    last_confirmed_at TEXT,
    last_declined_at TEXT,
    updated_at TEXT NOT NULL,
    user_persona_id TEXT,
    user_persona_key TEXT
        GENERATED ALWAYS AS (
            CASE WHEN user_persona_id IS NULL THEN 'n'
                 ELSE 'v:' || length(user_persona_id) || ':' || user_persona_id
            END
        ) VIRTUAL
);

INSERT INTO memory_consent_profile(
    _rowid, user_id, category, confirmed_count, declined_count,
    last_confirmed_at, last_declined_at, updated_at, user_persona_id
)
SELECT
    _rowid, user_id, category, confirmed_count, declined_count,
    last_confirmed_at, last_declined_at, updated_at, user_persona_id
FROM (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY
                user_id,
                CASE WHEN user_persona_id IS NULL THEN 'n' ELSE 'v:' || length(user_persona_id) || ':' || user_persona_id END,
                category
            ORDER BY updated_at DESC, _rowid DESC
        ) AS rn
    FROM memory_consent_profile_old
)
WHERE rn = 1;

DROP TABLE memory_consent_profile_old;

CREATE UNIQUE INDEX uq_memory_consent_profile_persona_category
    ON memory_consent_profile(user_id, user_persona_key, category);

ALTER TABLE graph_relationships RENAME TO graph_relationships_old;

CREATE TABLE graph_relationships (
    _rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    source_entity_id TEXT NOT NULL REFERENCES graph_entities(id) ON DELETE CASCADE,
    target_entity_id TEXT REFERENCES graph_entities(id) ON DELETE CASCADE,
    target_value_json TEXT,
    predicate TEXT NOT NULL,
    direction TEXT NOT NULL DEFAULT 'directed',
    scope TEXT NOT NULL DEFAULT 'chat' CHECK (scope IN ('chat', 'character', 'user')),
    workspace_id TEXT REFERENCES workspaces(id) ON DELETE CASCADE,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
    assistant_mode_id TEXT REFERENCES assistant_modes(id) ON DELETE CASCADE,
    confidence REAL NOT NULL DEFAULT 0.5,
    status TEXT NOT NULL DEFAULT 'active',
    valid_from TEXT,
    valid_to TEXT,
    privacy_level INTEGER NOT NULL DEFAULT 0,
    intimacy_boundary TEXT NOT NULL DEFAULT 'ordinary',
    intimacy_boundary_confidence REAL NOT NULL DEFAULT 0.0,
    supersedes_relationship_id TEXT REFERENCES graph_relationships(id) ON DELETE SET NULL,
    dedupe_key TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    user_persona_id TEXT,
    platform_id TEXT,
    character_id TEXT,
    sensitivity TEXT NOT NULL DEFAULT 'unknown' CHECK (sensitivity IN ('unknown', 'public', 'private', 'secret')),
    themes_json TEXT NOT NULL DEFAULT '[]',
    platform_locked INTEGER NOT NULL DEFAULT 0 CHECK (platform_locked IN (0, 1)),
    platform_id_lock TEXT,
    scope_canonical TEXT,
    CHECK (target_entity_id IS NOT NULL OR target_value_json IS NOT NULL),
    CHECK (predicate <> ''),
    CHECK (direction IN ('directed', 'symmetric')),
    CHECK (confidence BETWEEN 0.0 AND 1.0),
    CHECK (status IN ('active', 'review_required', 'superseded', 'conflicted', 'archived', 'deleted')),
    CHECK (privacy_level BETWEEN 0 AND 3),
    CHECK (
        intimacy_boundary IN ('ordinary', 'romantic_private', 'intimacy_private', 'intimacy_preference_private', 'intimacy_boundary', 'ambiguous_intimate', 'safety_blocked')
    ),
    CHECK (intimacy_boundary_confidence BETWEEN 0.0 AND 1.0)
);

INSERT INTO memory_redesign_phase11_report(table_name, disposition, affected_count, sample_ids_json, created_at)
SELECT
    'graph_relationships',
    'dropped_legacy_scope_rows',
    COUNT(*),
    COALESCE((SELECT json_group_array(id) FROM (
        SELECT id
        FROM graph_relationships_old
        WHERE COALESCE(scope_canonical, scope) IN ('workspace', 'legacy_workspace', 'assistant_mode', 'legacy_assistant_mode')
        ORDER BY id
        LIMIT 20
    )), '[]'),
    datetime('now')
FROM graph_relationships_old
WHERE COALESCE(scope_canonical, scope) IN ('workspace', 'legacy_workspace', 'assistant_mode', 'legacy_assistant_mode');

INSERT INTO graph_relationships(
    _rowid, id, user_id, source_entity_id, target_entity_id, target_value_json,
    predicate, direction, scope, workspace_id, conversation_id, assistant_mode_id,
    confidence, status, valid_from, valid_to, privacy_level, intimacy_boundary,
    intimacy_boundary_confidence, supersedes_relationship_id, dedupe_key,
    metadata_json, created_at, updated_at, user_persona_id, platform_id,
    character_id, sensitivity, themes_json, platform_locked, platform_id_lock,
    scope_canonical
)
SELECT
    _rowid, id, user_id, source_entity_id, target_entity_id, target_value_json,
    predicate, direction, canonical_scope, workspace_id, conversation_id, assistant_mode_id,
    confidence, status, valid_from, valid_to, privacy_level, intimacy_boundary,
    intimacy_boundary_confidence, supersedes_relationship_id, dedupe_key,
    metadata_json, created_at, updated_at, user_persona_id, platform_id,
    character_id, sensitivity, themes_json, platform_locked, platform_id_lock,
    canonical_scope
FROM (
    SELECT
        *,
        CASE
            WHEN COALESCE(scope_canonical, scope) IN ('conversation', 'ephemeral_session', 'chat') THEN 'chat'
            WHEN COALESCE(scope_canonical, scope) = 'character' THEN 'character'
            WHEN COALESCE(scope_canonical, scope) IN ('global_user', 'user') THEN 'user'
            WHEN COALESCE(scope_canonical, scope) IN ('workspace', 'legacy_workspace') THEN 'legacy_workspace'
            WHEN COALESCE(scope_canonical, scope) IN ('assistant_mode', 'legacy_assistant_mode') THEN 'legacy_assistant_mode'
            ELSE 'legacy_unknown'
        END AS canonical_scope
    FROM graph_relationships_old
)
WHERE canonical_scope IN ('chat', 'character', 'user');

DROP TABLE graph_relationships_old;

CREATE UNIQUE INDEX graph_relationships_user_dedupe_idx
    ON graph_relationships(user_id, dedupe_key);
CREATE INDEX graph_relationships_user_source_idx
    ON graph_relationships(user_id, source_entity_id, status, updated_at DESC, id ASC);
CREATE INDEX graph_relationships_user_target_idx
    ON graph_relationships(user_id, target_entity_id, status, updated_at DESC, id ASC);
CREATE INDEX graph_relationships_user_predicate_idx
    ON graph_relationships(user_id, predicate, status, updated_at DESC, id ASC);

DELETE FROM graph_relationship_sources
WHERE relationship_id NOT IN (SELECT id FROM graph_relationships)
   OR (memory_id IS NOT NULL AND memory_id NOT IN (SELECT id FROM memory_objects));

ALTER TABLE memory_links RENAME TO memory_links_old;

CREATE TABLE memory_links (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    src_memory_id TEXT NOT NULL REFERENCES memory_objects(id) ON DELETE CASCADE,
    dst_memory_id TEXT NOT NULL REFERENCES memory_objects(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL CHECK (
        relation_type IN ('supports', 'contradicts', 'depends_on', 'derived_from', 'supersedes', 'exception_to', 'about_topic', 'mentions_entity', 'led_to', 'reinforces', 'weakens')
    ),
    weight REAL NOT NULL DEFAULT 1.0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    user_persona_id TEXT,
    platform_id TEXT,
    character_id TEXT,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
    sensitivity TEXT NOT NULL DEFAULT 'unknown' CHECK (sensitivity IN ('unknown', 'public', 'private', 'secret')),
    platform_locked INTEGER NOT NULL DEFAULT 0 CHECK (platform_locked IN (0, 1)),
    platform_id_lock TEXT,
    policy_snapshot_json TEXT NOT NULL DEFAULT '{}'
);

INSERT INTO memory_redesign_phase11_report(table_name, disposition, affected_count, sample_ids_json, created_at)
SELECT
    'memory_links',
    'dropped_legacy_or_cross_namespace_rows',
    COUNT(*),
    COALESCE((SELECT json_group_array(id) FROM (
        SELECT ml.id
        FROM memory_links_old AS ml
        LEFT JOIN memory_objects AS src ON src.id = ml.src_memory_id
        LEFT JOIN memory_objects AS dst ON dst.id = ml.dst_memory_id
        WHERE ml.relation_type IN ('applies_in_mode', 'belongs_to_workspace')
           OR src.id IS NULL
           OR dst.id IS NULL
           OR src.user_persona_id IS NOT dst.user_persona_id
           OR COALESCE(src.platform_id, '') != COALESCE(dst.platform_id, '')
           OR src.character_id IS NOT dst.character_id
           OR src.conversation_id IS NOT dst.conversation_id
        ORDER BY ml.id
        LIMIT 20
    )), '[]'),
    datetime('now')
FROM memory_links_old AS ml
LEFT JOIN memory_objects AS src ON src.id = ml.src_memory_id
LEFT JOIN memory_objects AS dst ON dst.id = ml.dst_memory_id
WHERE ml.relation_type IN ('applies_in_mode', 'belongs_to_workspace')
   OR src.id IS NULL
   OR dst.id IS NULL
   OR src.user_persona_id IS NOT dst.user_persona_id
   OR COALESCE(src.platform_id, '') != COALESCE(dst.platform_id, '')
   OR src.character_id IS NOT dst.character_id
   OR src.conversation_id IS NOT dst.conversation_id;

INSERT INTO memory_links(
    id, user_id, src_memory_id, dst_memory_id, relation_type, weight,
    metadata_json, created_at, user_persona_id, platform_id, character_id,
    conversation_id, sensitivity, platform_locked, platform_id_lock,
    policy_snapshot_json
)
SELECT
    ml.id,
    ml.user_id,
    ml.src_memory_id,
    ml.dst_memory_id,
    ml.relation_type,
    ml.weight,
    ml.metadata_json,
    ml.created_at,
    src.user_persona_id,
    src.platform_id,
    src.character_id,
    src.conversation_id,
    CASE
        WHEN src.sensitivity = 'secret' OR dst.sensitivity = 'secret' THEN 'secret'
        WHEN src.sensitivity = 'private' OR dst.sensitivity = 'private' THEN 'private'
        WHEN src.sensitivity = 'public' OR dst.sensitivity = 'public' THEN 'public'
        ELSE 'unknown'
    END,
    MAX(src.platform_locked, dst.platform_locked),
    COALESCE(src.platform_id_lock, dst.platform_id_lock),
    json_object(
        'phase11', 1,
        'src_scope', src.scope_canonical,
        'dst_scope', dst.scope_canonical
    )
FROM memory_links_old AS ml
JOIN memory_objects AS src ON src.id = ml.src_memory_id
JOIN memory_objects AS dst ON dst.id = ml.dst_memory_id
WHERE ml.relation_type NOT IN ('applies_in_mode', 'belongs_to_workspace')
  AND src.user_persona_id IS dst.user_persona_id
  AND COALESCE(src.platform_id, '') = COALESCE(dst.platform_id, '')
  AND src.character_id IS dst.character_id
  AND src.conversation_id IS dst.conversation_id;

DROP TABLE memory_links_old;

CREATE INDEX idx_memory_links_user_src
    ON memory_links(user_id, src_memory_id);
CREATE INDEX idx_memory_links_dst
    ON memory_links(dst_memory_id);
CREATE INDEX idx_memory_links_relation
    ON memory_links(relation_type);
CREATE INDEX idx_memory_links_namespace
    ON memory_links(user_id, user_persona_id, character_id, conversation_id, platform_id);

ALTER TABLE conversations RENAME TO conversations_old;

CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    workspace_id TEXT REFERENCES workspaces(id) ON DELETE SET NULL,
    assistant_mode_id TEXT REFERENCES assistant_modes(id) ON DELETE RESTRICT,
    title TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    metadata_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    temporary INTEGER NOT NULL DEFAULT 0 CHECK (temporary IN (0, 1)),
    temporary_ttl_seconds INTEGER CHECK (temporary_ttl_seconds IS NULL OR temporary_ttl_seconds > 0),
    purge_on_close INTEGER NOT NULL DEFAULT 0 CHECK (purge_on_close IN (0, 1)),
    last_activity_at TEXT,
    closed_at TEXT,
    isolated_mode INTEGER NOT NULL DEFAULT 0 CHECK (isolated_mode IN (0, 1)),
    user_persona_id TEXT,
    platform_id TEXT,
    character_id TEXT,
    mode TEXT,
    incognito INTEGER NOT NULL DEFAULT 0 CHECK (incognito IN (0, 1))
);

INSERT INTO conversations
SELECT * FROM conversations_old;

DROP TABLE conversations_old;

CREATE INDEX idx_conversations_user
    ON conversations(user_id);
CREATE INDEX idx_conversations_user_status_temporary
    ON conversations(user_id, status, temporary, updated_at DESC, id ASC);
CREATE INDEX idx_conversations_temporary_ttl
    ON conversations(temporary, status, last_activity_at)
    WHERE temporary = 1 AND temporary_ttl_seconds IS NOT NULL;
CREATE INDEX idx_conversations_identity
    ON conversations(user_id, user_persona_id, character_id, platform_id, status, updated_at DESC);
CREATE INDEX idx_conversations_incognito_v2
    ON conversations(user_id, incognito, updated_at DESC, id ASC);

ALTER TABLE conversation_activity_stats RENAME TO conversation_activity_stats_old;

CREATE TABLE conversation_activity_stats (
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    workspace_id TEXT,
    assistant_mode_id TEXT,
    timezone TEXT NOT NULL DEFAULT 'UTC',
    first_message_at TEXT,
    last_message_at TEXT,
    last_user_message_at TEXT,
    message_count INTEGER NOT NULL DEFAULT 0,
    user_message_count INTEGER NOT NULL DEFAULT 0,
    assistant_message_count INTEGER NOT NULL DEFAULT 0,
    retrieval_count INTEGER NOT NULL DEFAULT 0,
    active_day_count INTEGER NOT NULL DEFAULT 0,
    recent_1d_message_count INTEGER NOT NULL DEFAULT 0,
    recent_7d_message_count INTEGER NOT NULL DEFAULT 0,
    recent_30d_message_count INTEGER NOT NULL DEFAULT 0,
    weekday_histogram_json TEXT NOT NULL DEFAULT '[]',
    hour_histogram_json TEXT NOT NULL DEFAULT '[]',
    hour_of_week_histogram_json TEXT NOT NULL DEFAULT '[]',
    return_interval_histogram_json TEXT NOT NULL DEFAULT '[]',
    avg_return_interval_minutes REAL,
    median_return_interval_minutes REAL,
    p90_return_interval_minutes REAL,
    main_thread_score REAL NOT NULL DEFAULT 0.0,
    likely_soon_score REAL NOT NULL DEFAULT 0.0,
    return_habit_confidence REAL NOT NULL DEFAULT 0.0,
    schedule_pattern_kind TEXT NOT NULL DEFAULT 'inactive',
    activity_version INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT NOT NULL,
    user_persona_id TEXT,
    platform_id TEXT,
    character_id TEXT,
    incognito INTEGER NOT NULL DEFAULT 0 CHECK (incognito IN (0, 1)),
    remember_across_chats INTEGER NOT NULL DEFAULT 1 CHECK (remember_across_chats IN (0, 1)),
    remember_across_devices INTEGER NOT NULL DEFAULT 1 CHECK (remember_across_devices IN (0, 1)),
    effective_policy_hash TEXT,
    PRIMARY KEY (user_id, conversation_id)
);

INSERT INTO conversation_activity_stats
SELECT
    cas.user_id,
    cas.conversation_id,
    cas.workspace_id,
    cas.assistant_mode_id,
    cas.timezone,
    cas.first_message_at,
    cas.last_message_at,
    cas.last_user_message_at,
    cas.message_count,
    cas.user_message_count,
    cas.assistant_message_count,
    cas.retrieval_count,
    cas.active_day_count,
    cas.recent_1d_message_count,
    cas.recent_7d_message_count,
    cas.recent_30d_message_count,
    cas.weekday_histogram_json,
    cas.hour_histogram_json,
    cas.hour_of_week_histogram_json,
    cas.return_interval_histogram_json,
    cas.avg_return_interval_minutes,
    cas.median_return_interval_minutes,
    cas.p90_return_interval_minutes,
    cas.main_thread_score,
    cas.likely_soon_score,
    cas.return_habit_confidence,
    cas.schedule_pattern_kind,
    cas.activity_version,
    cas.updated_at,
    COALESCE(cas.user_persona_id, c.user_persona_id),
    COALESCE(cas.platform_id, c.platform_id),
    COALESCE(cas.character_id, c.character_id),
    COALESCE(cas.incognito, c.incognito, 0),
    COALESCE(cas.remember_across_chats, u.remember_across_chats, 1),
    COALESCE(cas.remember_across_devices, u.remember_across_devices, 1),
    cas.effective_policy_hash
FROM conversation_activity_stats_old AS cas
JOIN conversations AS c ON c.id = cas.conversation_id AND c.user_id = cas.user_id
JOIN users AS u ON u.id = cas.user_id;

DROP TABLE conversation_activity_stats_old;

CREATE INDEX idx_conversation_activity_user_hot
    ON conversation_activity_stats(user_id, likely_soon_score DESC, main_thread_score DESC, last_message_at DESC, conversation_id ASC);
CREATE INDEX idx_conversation_activity_identity
    ON conversation_activity_stats(
        user_id, user_persona_id, character_id, platform_id, incognito,
        likely_soon_score DESC, last_message_at DESC, conversation_id ASC
    );

ALTER TABLE retrieval_events RENAME TO retrieval_events_old;

CREATE TABLE retrieval_events (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    request_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    response_message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
    assistant_mode_id TEXT,
    retrieval_plan_json TEXT NOT NULL,
    selected_memory_ids_json TEXT NOT NULL,
    context_view_json TEXT NOT NULL,
    outcome_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    user_persona_id TEXT,
    platform_id TEXT,
    character_id TEXT,
    mode TEXT,
    incognito INTEGER NOT NULL DEFAULT 0 CHECK (incognito IN (0, 1)),
    remember_across_chats INTEGER NOT NULL DEFAULT 1 CHECK (remember_across_chats IN (0, 1)),
    remember_across_devices INTEGER NOT NULL DEFAULT 1 CHECK (remember_across_devices IN (0, 1))
);

INSERT INTO retrieval_events
SELECT
    re.id,
    re.user_id,
    re.conversation_id,
    re.request_message_id,
    re.response_message_id,
    re.assistant_mode_id,
    re.retrieval_plan_json,
    re.selected_memory_ids_json,
    re.context_view_json,
    re.outcome_json,
    re.created_at,
    COALESCE(re.user_persona_id, c.user_persona_id),
    COALESCE(re.platform_id, c.platform_id),
    COALESCE(re.character_id, c.character_id),
    COALESCE(re.mode, c.mode, re.assistant_mode_id),
    COALESCE(re.incognito, c.incognito, 0),
    COALESCE(re.remember_across_chats, u.remember_across_chats, 1),
    COALESCE(re.remember_across_devices, u.remember_across_devices, 1)
FROM retrieval_events_old AS re
JOIN conversations AS c ON c.id = re.conversation_id AND c.user_id = re.user_id
JOIN users AS u ON u.id = re.user_id;

DROP TABLE retrieval_events_old;

CREATE INDEX idx_retrieval_events_user
    ON retrieval_events(user_id, created_at);
CREATE INDEX idx_retrieval_events_conversation
    ON retrieval_events(conversation_id);

PRAGMA legacy_alter_table = OFF;

DROP TABLE phase11_dropped_memory_ids;
