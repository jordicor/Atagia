-- atagia:foreign_keys_off

PRAGMA legacy_alter_table = ON;

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
    incognito INTEGER NOT NULL DEFAULT 0 CHECK (incognito IN (0, 1)),
    active_presence_id TEXT,
    active_space_id TEXT,
    active_mind_id TEXT,
    mind_topology TEXT NOT NULL DEFAULT 'unimind' CHECK (
        mind_topology IN ('unimind', 'multi_mind', 'ojocentauri')
    ),
    active_embodiment_id TEXT,
    active_realm_id TEXT
);

INSERT INTO conversations(
    id,
    user_id,
    workspace_id,
    assistant_mode_id,
    title,
    status,
    metadata_json,
    created_at,
    updated_at,
    temporary,
    temporary_ttl_seconds,
    purge_on_close,
    last_activity_at,
    closed_at,
    isolated_mode,
    user_persona_id,
    platform_id,
    character_id,
    mode,
    incognito,
    active_presence_id,
    active_space_id,
    active_mind_id,
    mind_topology,
    active_embodiment_id,
    active_realm_id
)
SELECT
    id,
    user_id,
    workspace_id,
    assistant_mode_id,
    title,
    status,
    metadata_json,
    created_at,
    updated_at,
    temporary,
    temporary_ttl_seconds,
    purge_on_close,
    last_activity_at,
    closed_at,
    isolated_mode,
    user_persona_id,
    platform_id,
    character_id,
    mode,
    incognito,
    active_presence_id,
    active_space_id,
    active_mind_id,
    mind_topology,
    active_embodiment_id,
    active_realm_id
FROM conversations_old;

DROP TABLE conversations_old;

CREATE INDEX IF NOT EXISTS idx_conversations_user
    ON conversations(user_id);

CREATE INDEX IF NOT EXISTS idx_conversations_user_status_temporary
    ON conversations(user_id, status, temporary, updated_at DESC, id ASC);

CREATE INDEX IF NOT EXISTS idx_conversations_temporary_ttl
    ON conversations(temporary, status, last_activity_at)
    WHERE temporary = 1 AND temporary_ttl_seconds IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_conversations_identity
    ON conversations(user_id, user_persona_id, character_id, platform_id, status, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_conversations_incognito_v2
    ON conversations(user_id, incognito, updated_at DESC, id ASC);

CREATE INDEX IF NOT EXISTS idx_conversations_active_presence
    ON conversations(user_id, active_presence_id, status, updated_at DESC)
    WHERE active_presence_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_conversations_active_space
    ON conversations(user_id, active_space_id, status, updated_at DESC)
    WHERE active_space_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS conversations_user_active_mind_idx
    ON conversations(user_id, active_mind_id, status, updated_at DESC)
    WHERE active_mind_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS conversations_user_active_embodiment_idx
    ON conversations(user_id, active_embodiment_id, status, updated_at DESC)
    WHERE active_embodiment_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS conversations_user_active_realm_idx
    ON conversations(user_id, active_realm_id, status, updated_at DESC)
    WHERE active_realm_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS overseer_grants (
    owner_user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    overseer_mind_id TEXT NOT NULL,
    target_kind TEXT NOT NULL CHECK (target_kind IN ('mind', 'space', 'realm')),
    target_id TEXT NOT NULL,
    grant_kind TEXT NOT NULL CHECK (
        grant_kind IN ('read', 'summarize', 'coordinate', 'audit', 'rescope')
    ),
    visibility TEXT NOT NULL DEFAULT 'attributed' CHECK (
        visibility IN ('attributed', 'applicable')
    ),
    expires_at TEXT,
    revoked_at TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (owner_user_id, overseer_mind_id, target_kind, target_id, grant_kind),
    FOREIGN KEY (owner_user_id, overseer_mind_id)
        REFERENCES minds(owner_user_id, id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS overseer_grants_active_idx
    ON overseer_grants(
        owner_user_id,
        overseer_mind_id,
        target_kind,
        target_id,
        grant_kind,
        revoked_at,
        expires_at
    );

PRAGMA legacy_alter_table = OFF;
