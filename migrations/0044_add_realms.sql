CREATE TABLE IF NOT EXISTS realms (
    id TEXT NOT NULL,
    owner_user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    display_name TEXT,
    source_kind TEXT NOT NULL,
    source_id TEXT NOT NULL,
    cross_realm_mode TEXT NOT NULL DEFAULT 'none' CHECK (
        cross_realm_mode IN ('none', 'attributed', 'applicable')
    ),
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (owner_user_id, id),
    UNIQUE (owner_user_id, source_kind, source_id)
);

CREATE TABLE IF NOT EXISTS realm_bridges (
    owner_user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    source_realm_id TEXT NOT NULL,
    target_realm_id TEXT NOT NULL,
    cross_realm_mode TEXT NOT NULL CHECK (
        cross_realm_mode IN ('attributed', 'applicable')
    ),
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (owner_user_id, source_realm_id, target_realm_id),
    FOREIGN KEY (owner_user_id, source_realm_id)
        REFERENCES realms(owner_user_id, id) ON DELETE CASCADE,
    FOREIGN KEY (owner_user_id, target_realm_id)
        REFERENCES realms(owner_user_id, id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS realms_owner_updated_idx
    ON realms(owner_user_id, updated_at DESC, id ASC);

CREATE INDEX IF NOT EXISTS realm_bridges_owner_source_idx
    ON realm_bridges(owner_user_id, source_realm_id, cross_realm_mode, target_realm_id);

ALTER TABLE conversations ADD COLUMN active_realm_id TEXT;

ALTER TABLE messages ADD COLUMN active_realm_id TEXT;

ALTER TABLE memory_objects ADD COLUMN realm_id TEXT;

ALTER TABLE artifacts ADD COLUMN realm_id TEXT;

ALTER TABLE verbatim_pins ADD COLUMN realm_id TEXT;

ALTER TABLE contract_dimensions_current ADD COLUMN realm_id TEXT;

ALTER TABLE contract_dimensions_current
    ADD COLUMN realm_key TEXT
        GENERATED ALWAYS AS (
            CASE WHEN realm_id IS NULL THEN 'n'
                 ELSE 'v:' || length(realm_id) || ':' || realm_id
            END
        ) VIRTUAL;

UPDATE messages
SET active_realm_id = (
    SELECT conversations.active_realm_id
    FROM conversations
    WHERE conversations.id = messages.conversation_id
)
WHERE active_realm_id IS NULL
  AND EXISTS (
      SELECT 1
      FROM conversations
      WHERE conversations.id = messages.conversation_id
        AND conversations.active_realm_id IS NOT NULL
  );

UPDATE artifacts
SET realm_id = (
    SELECT messages.active_realm_id
    FROM messages
    WHERE messages.id = artifacts.message_id
)
WHERE message_id IS NOT NULL
  AND EXISTS (
      SELECT 1
      FROM messages
      WHERE messages.id = artifacts.message_id
        AND messages.active_realm_id IS NOT NULL
  );

UPDATE verbatim_pins
SET realm_id = (
    SELECT memory_objects.realm_id
    FROM memory_objects
    WHERE memory_objects.id = verbatim_pins.target_id
      AND memory_objects.user_id = verbatim_pins.user_id
)
WHERE target_kind = 'memory_object'
  AND EXISTS (
      SELECT 1
      FROM memory_objects
      WHERE memory_objects.id = verbatim_pins.target_id
        AND memory_objects.user_id = verbatim_pins.user_id
        AND memory_objects.realm_id IS NOT NULL
  );

UPDATE verbatim_pins
SET realm_id = (
    SELECT messages.active_realm_id
    FROM messages
    JOIN conversations ON conversations.id = messages.conversation_id
    WHERE messages.id = verbatim_pins.target_id
      AND conversations.user_id = verbatim_pins.user_id
)
WHERE target_kind IN ('message', 'text_span')
  AND realm_id IS NULL
  AND EXISTS (
      SELECT 1
      FROM messages
      JOIN conversations ON conversations.id = messages.conversation_id
      WHERE messages.id = verbatim_pins.target_id
        AND conversations.user_id = verbatim_pins.user_id
        AND messages.active_realm_id IS NOT NULL
  );

UPDATE verbatim_pins
SET realm_id = (
    SELECT conversations.active_realm_id
    FROM conversations
    WHERE conversations.id = verbatim_pins.conversation_id
      AND conversations.user_id = verbatim_pins.user_id
)
WHERE realm_id IS NULL
  AND conversation_id IS NOT NULL
  AND EXISTS (
      SELECT 1
      FROM conversations
      WHERE conversations.id = verbatim_pins.conversation_id
        AND conversations.user_id = verbatim_pins.user_id
        AND conversations.active_realm_id IS NOT NULL
  );

UPDATE contract_dimensions_current
SET realm_id = (
    SELECT memory_objects.realm_id
    FROM memory_objects
    WHERE memory_objects.id = contract_dimensions_current.source_memory_id
      AND memory_objects.user_id = contract_dimensions_current.user_id
)
WHERE source_memory_id IS NOT NULL;

DROP INDEX IF EXISTS uq_contract_dimensions_current;

CREATE UNIQUE INDEX IF NOT EXISTS uq_contract_dimensions_current
    ON contract_dimensions_current(
        user_id,
        user_persona_key,
        character_key,
        conversation_key,
        scope_canonical_key,
        space_key,
        memory_owner_key,
        embodiment_key,
        realm_key,
        dimension_name
    );

CREATE INDEX IF NOT EXISTS conversations_user_active_realm_idx
    ON conversations(user_id, active_realm_id, status, updated_at DESC)
    WHERE active_realm_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS messages_conversation_active_realm_idx
    ON messages(conversation_id, active_realm_id, seq ASC)
    WHERE active_realm_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS memory_objects_user_realm_idx
    ON memory_objects(user_id, realm_id, status, updated_at DESC, id ASC)
    WHERE realm_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS artifacts_user_realm_idx
    ON artifacts(user_id, realm_id, status, updated_at DESC, id ASC)
    WHERE realm_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS verbatim_pins_user_realm_status_idx
    ON verbatim_pins(user_id, realm_id, status, updated_at DESC, id ASC)
    WHERE realm_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS contract_dimensions_current_realm_idx
    ON contract_dimensions_current(user_id, realm_id, scope_canonical, updated_at DESC)
    WHERE realm_id IS NOT NULL;
