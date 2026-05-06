-- Phase 10: secondary retrieval surfaces become namespace/policy-aware.

ALTER TABLE contract_dimensions_current
    ADD COLUMN scope_canonical_key TEXT
        GENERATED ALWAYS AS (
            CASE
                WHEN scope_canonical IS NOT NULL THEN scope_canonical
                WHEN scope = 'global_user' THEN 'user'
                WHEN scope = 'assistant_mode' THEN 'legacy_assistant_mode'
                WHEN scope = 'workspace' THEN 'legacy_workspace'
                WHEN scope = 'conversation' THEN 'chat'
                WHEN scope = 'ephemeral_session' THEN 'chat'
                ELSE scope
            END
        ) VIRTUAL;

UPDATE contract_dimensions_current
SET scope_canonical = CASE scope
        WHEN 'global_user' THEN 'user'
        WHEN 'assistant_mode' THEN 'legacy_assistant_mode'
        WHEN 'workspace' THEN 'legacy_workspace'
        WHEN 'conversation' THEN 'chat'
        WHEN 'ephemeral_session' THEN 'chat'
        ELSE scope
    END
WHERE scope_canonical IS NULL;

DROP INDEX IF EXISTS uq_contract_dimensions_current;

CREATE UNIQUE INDEX IF NOT EXISTS uq_contract_dimensions_current
    ON contract_dimensions_current(
        user_id,
        user_persona_key,
        character_key,
        conversation_key,
        scope_canonical_key,
        dimension_name
    );

ALTER TABLE artifacts ADD COLUMN user_persona_id TEXT;
ALTER TABLE artifacts ADD COLUMN platform_id TEXT;
ALTER TABLE artifacts ADD COLUMN character_id TEXT;
ALTER TABLE artifacts
    ADD COLUMN sensitivity TEXT NOT NULL DEFAULT 'unknown'
        CHECK (sensitivity IN ('unknown', 'public', 'private', 'secret'));
ALTER TABLE artifacts
    ADD COLUMN themes_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE artifacts
    ADD COLUMN platform_locked INTEGER NOT NULL DEFAULT 0
        CHECK (platform_locked IN (0, 1));
ALTER TABLE artifacts ADD COLUMN platform_id_lock TEXT;
ALTER TABLE artifacts ADD COLUMN scope_canonical TEXT;
ALTER TABLE artifacts
    ADD COLUMN incognito_snapshot INTEGER NOT NULL DEFAULT 0
        CHECK (incognito_snapshot IN (0, 1));
ALTER TABLE artifacts
    ADD COLUMN remember_across_chats_snapshot INTEGER NOT NULL DEFAULT 1
        CHECK (remember_across_chats_snapshot IN (0, 1));
ALTER TABLE artifacts
    ADD COLUMN remember_across_devices_snapshot INTEGER NOT NULL DEFAULT 1
        CHECK (remember_across_devices_snapshot IN (0, 1));
ALTER TABLE artifacts
    ADD COLUMN policy_snapshot_json TEXT NOT NULL DEFAULT '{}';

UPDATE artifacts
SET user_persona_id = (
        SELECT conversations.user_persona_id
        FROM conversations
        WHERE conversations.id = artifacts.conversation_id
          AND conversations.user_id = artifacts.user_id
    ),
    platform_id = COALESCE((
        SELECT conversations.platform_id
        FROM conversations
        WHERE conversations.id = artifacts.conversation_id
          AND conversations.user_id = artifacts.user_id
    ), 'default'),
    character_id = COALESCE((
        SELECT conversations.character_id
        FROM conversations
        WHERE conversations.id = artifacts.conversation_id
          AND conversations.user_id = artifacts.user_id
    ), workspace_id),
    sensitivity = CASE
        WHEN privacy_level >= 3 THEN 'secret'
        WHEN privacy_level >= 2 THEN 'private'
        WHEN intimacy_boundary IS NOT NULL AND intimacy_boundary != 'ordinary' THEN 'private'
        ELSE 'public'
    END,
    platform_locked = COALESCE((
        SELECT CASE WHEN users.remember_across_devices = 0 THEN 1 ELSE 0 END
        FROM users
        WHERE users.id = artifacts.user_id
    ), 0),
    platform_id_lock = (
        SELECT CASE
            WHEN users.remember_across_devices = 0
                THEN COALESCE((
                    SELECT conversations.platform_id
                    FROM conversations
                    WHERE conversations.id = artifacts.conversation_id
                      AND conversations.user_id = artifacts.user_id
                ), 'default')
            ELSE NULL
        END
        FROM users
        WHERE users.id = artifacts.user_id
    ),
    scope_canonical = CASE
        WHEN conversation_id IS NOT NULL THEN 'chat'
        WHEN workspace_id IS NOT NULL THEN 'character'
        ELSE 'user'
    END,
    incognito_snapshot = COALESCE((
        SELECT conversations.incognito
        FROM conversations
        WHERE conversations.id = artifacts.conversation_id
          AND conversations.user_id = artifacts.user_id
    ), 0),
    remember_across_chats_snapshot = COALESCE((
        SELECT users.remember_across_chats
        FROM users
        WHERE users.id = artifacts.user_id
    ), 1),
    remember_across_devices_snapshot = COALESCE((
        SELECT users.remember_across_devices
        FROM users
        WHERE users.id = artifacts.user_id
    ), 1),
    policy_snapshot_json = json_object(
        'user_persona_id', (
            SELECT conversations.user_persona_id
            FROM conversations
            WHERE conversations.id = artifacts.conversation_id
              AND conversations.user_id = artifacts.user_id
        ),
        'platform_id', COALESCE((
            SELECT conversations.platform_id
            FROM conversations
            WHERE conversations.id = artifacts.conversation_id
              AND conversations.user_id = artifacts.user_id
        ), 'default'),
        'character_id', COALESCE((
            SELECT conversations.character_id
            FROM conversations
            WHERE conversations.id = artifacts.conversation_id
              AND conversations.user_id = artifacts.user_id
        ), workspace_id),
        'conversation_id', conversation_id,
        'incognito', COALESCE((
            SELECT conversations.incognito
            FROM conversations
            WHERE conversations.id = artifacts.conversation_id
              AND conversations.user_id = artifacts.user_id
        ), 0),
        'remember_across_chats', COALESCE((
            SELECT users.remember_across_chats
            FROM users
            WHERE users.id = artifacts.user_id
        ), 1),
        'remember_across_devices', COALESCE((
            SELECT users.remember_across_devices
            FROM users
            WHERE users.id = artifacts.user_id
        ), 1),
        'intended_scope', CASE
            WHEN conversation_id IS NOT NULL THEN 'chat'
            WHEN workspace_id IS NOT NULL THEN 'character'
            ELSE 'user'
        END,
        'platform_locked', COALESCE((
            SELECT CASE WHEN users.remember_across_devices = 0 THEN 1 ELSE 0 END
            FROM users
            WHERE users.id = artifacts.user_id
        ), 0),
        'platform_id_lock', (
            SELECT CASE
                WHEN users.remember_across_devices = 0
                    THEN COALESCE((
                        SELECT conversations.platform_id
                        FROM conversations
                        WHERE conversations.id = artifacts.conversation_id
                          AND conversations.user_id = artifacts.user_id
                    ), 'default')
                ELSE NULL
            END
            FROM users
            WHERE users.id = artifacts.user_id
        )
    );

CREATE INDEX IF NOT EXISTS artifacts_user_namespace_idx
    ON artifacts(user_id, user_persona_id, character_id, scope_canonical, status, updated_at DESC);

CREATE INDEX IF NOT EXISTS artifacts_user_platform_lock_idx
    ON artifacts(user_id, platform_locked, platform_id_lock, status);
