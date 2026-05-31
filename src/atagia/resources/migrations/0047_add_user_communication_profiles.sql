CREATE TABLE IF NOT EXISTS user_communication_profiles (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    profile_kind TEXT NOT NULL,
    scope_canonical TEXT NOT NULL CHECK (scope_canonical IN ('chat', 'character', 'user')),
    workspace_id TEXT REFERENCES workspaces(id) ON DELETE SET NULL,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
    assistant_mode_id TEXT REFERENCES assistant_modes(id) ON DELETE SET NULL,
    user_persona_id TEXT,
    platform_id TEXT,
    character_id TEXT,
    subject_presence_id TEXT,
    space_id TEXT,
    space_boundary_mode TEXT,
    memory_owner_id TEXT,
    source_mind_id TEXT,
    embodiment_id TEXT,
    realm_id TEXT,
    profile_json TEXT NOT NULL,
    source_refs_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'deleted')),
    stale INTEGER NOT NULL DEFAULT 0 CHECK (stale IN (0, 1)),
    stale_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    user_persona_key TEXT
        GENERATED ALWAYS AS (
            CASE WHEN user_persona_id IS NULL THEN 'n'
                 ELSE 'v:' || length(user_persona_id) || ':' || user_persona_id
            END
        ) VIRTUAL,
    platform_key TEXT
        GENERATED ALWAYS AS (
            CASE WHEN platform_id IS NULL THEN 'n'
                 ELSE 'v:' || length(platform_id) || ':' || platform_id
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
    space_key TEXT
        GENERATED ALWAYS AS (
            CASE WHEN space_id IS NULL THEN 'n'
                 ELSE 'v:' || length(space_id) || ':' || space_id
            END
        ) VIRTUAL,
    memory_owner_key TEXT
        GENERATED ALWAYS AS (
            CASE WHEN memory_owner_id IS NULL THEN 'n'
                 ELSE 'v:' || length(memory_owner_id) || ':' || memory_owner_id
            END
        ) VIRTUAL,
    embodiment_key TEXT
        GENERATED ALWAYS AS (
            CASE WHEN embodiment_id IS NULL THEN 'n'
                 ELSE 'v:' || length(embodiment_id) || ':' || embodiment_id
            END
        ) VIRTUAL,
    realm_key TEXT
        GENERATED ALWAYS AS (
            CASE WHEN realm_id IS NULL THEN 'n'
                 ELSE 'v:' || length(realm_id) || ':' || realm_id
            END
        ) VIRTUAL
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_user_communication_profiles_target
    ON user_communication_profiles(
        user_id,
        profile_kind,
        scope_canonical,
        user_persona_key,
        platform_key,
        character_key,
        conversation_key,
        space_key,
        memory_owner_key,
        embodiment_key,
        realm_key
    );

CREATE INDEX IF NOT EXISTS idx_user_communication_profiles_lookup
    ON user_communication_profiles(
        user_id,
        profile_kind,
        status,
        stale,
        scope_canonical,
        updated_at DESC
    );

CREATE INDEX IF NOT EXISTS idx_user_communication_profiles_conversation
    ON user_communication_profiles(user_id, conversation_id, status);
