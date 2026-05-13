CREATE TABLE IF NOT EXISTS minds (
    id TEXT NOT NULL,
    owner_user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    kind TEXT NOT NULL DEFAULT 'unknown' CHECK (
        kind IN ('human', 'owned_ai', 'owned_facet', 'external_actor', 'overseer', 'unknown')
    ),
    display_name TEXT,
    source_kind TEXT NOT NULL,
    source_id TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (owner_user_id, id),
    UNIQUE (owner_user_id, source_kind, source_id)
);

CREATE INDEX IF NOT EXISTS minds_owner_kind_idx
    ON minds(owner_user_id, kind, updated_at DESC, id ASC);

ALTER TABLE conversations ADD COLUMN active_mind_id TEXT;

ALTER TABLE conversations ADD COLUMN mind_topology TEXT NOT NULL DEFAULT 'unimind' CHECK (
    mind_topology IN ('unimind', 'multi_mind')
);

ALTER TABLE messages ADD COLUMN active_mind_id TEXT;

ALTER TABLE messages ADD COLUMN source_mind_id TEXT;

ALTER TABLE memory_objects ADD COLUMN memory_owner_id TEXT;

ALTER TABLE memory_objects ADD COLUMN source_mind_id TEXT;

ALTER TABLE artifacts ADD COLUMN memory_owner_id TEXT;

ALTER TABLE artifacts ADD COLUMN source_mind_id TEXT;

ALTER TABLE verbatim_pins ADD COLUMN memory_owner_id TEXT;

ALTER TABLE verbatim_pins ADD COLUMN source_mind_id TEXT;

ALTER TABLE contract_dimensions_current ADD COLUMN memory_owner_id TEXT;

ALTER TABLE contract_dimensions_current ADD COLUMN source_mind_id TEXT;

ALTER TABLE contract_dimensions_current
    ADD COLUMN memory_owner_key TEXT
        GENERATED ALWAYS AS (
            CASE WHEN memory_owner_id IS NULL THEN 'n'
                 ELSE 'v:' || length(memory_owner_id) || ':' || memory_owner_id
            END
        ) VIRTUAL;

UPDATE messages
SET
    active_mind_id = (
        SELECT conversations.active_mind_id
        FROM conversations
        WHERE conversations.id = messages.conversation_id
    ),
    source_mind_id = (
        SELECT conversations.active_mind_id
        FROM conversations
        WHERE conversations.id = messages.conversation_id
    )
WHERE active_mind_id IS NULL
  AND EXISTS (
      SELECT 1
      FROM conversations
      WHERE conversations.id = messages.conversation_id
        AND conversations.active_mind_id IS NOT NULL
  );

UPDATE artifacts
SET
    memory_owner_id = (
        SELECT messages.active_mind_id
        FROM messages
        WHERE messages.id = artifacts.message_id
    ),
    source_mind_id = (
        SELECT messages.source_mind_id
        FROM messages
        WHERE messages.id = artifacts.message_id
    )
WHERE message_id IS NOT NULL
  AND EXISTS (
      SELECT 1
      FROM messages
      WHERE messages.id = artifacts.message_id
        AND messages.active_mind_id IS NOT NULL
  );

UPDATE verbatim_pins
SET
    memory_owner_id = (
        SELECT memory_objects.memory_owner_id
        FROM memory_objects
        WHERE memory_objects.id = verbatim_pins.target_id
          AND memory_objects.user_id = verbatim_pins.user_id
    ),
    source_mind_id = (
        SELECT memory_objects.source_mind_id
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
        AND memory_objects.memory_owner_id IS NOT NULL
  );

UPDATE verbatim_pins
SET
    memory_owner_id = (
        SELECT messages.active_mind_id
        FROM messages
        JOIN conversations ON conversations.id = messages.conversation_id
        WHERE messages.id = verbatim_pins.target_id
          AND conversations.user_id = verbatim_pins.user_id
    ),
    source_mind_id = (
        SELECT messages.source_mind_id
        FROM messages
        JOIN conversations ON conversations.id = messages.conversation_id
        WHERE messages.id = verbatim_pins.target_id
          AND conversations.user_id = verbatim_pins.user_id
    )
WHERE target_kind IN ('message', 'text_span')
  AND memory_owner_id IS NULL
  AND EXISTS (
      SELECT 1
      FROM messages
      JOIN conversations ON conversations.id = messages.conversation_id
      WHERE messages.id = verbatim_pins.target_id
        AND conversations.user_id = verbatim_pins.user_id
        AND messages.active_mind_id IS NOT NULL
  );

UPDATE verbatim_pins
SET
    memory_owner_id = (
        SELECT conversations.active_mind_id
        FROM conversations
        WHERE conversations.id = verbatim_pins.conversation_id
          AND conversations.user_id = verbatim_pins.user_id
    ),
    source_mind_id = (
        SELECT conversations.active_mind_id
        FROM conversations
        WHERE conversations.id = verbatim_pins.conversation_id
          AND conversations.user_id = verbatim_pins.user_id
    )
WHERE memory_owner_id IS NULL
  AND conversation_id IS NOT NULL
  AND EXISTS (
      SELECT 1
      FROM conversations
      WHERE conversations.id = verbatim_pins.conversation_id
        AND conversations.user_id = verbatim_pins.user_id
        AND conversations.active_mind_id IS NOT NULL
  );

UPDATE contract_dimensions_current
SET
    memory_owner_id = (
        SELECT memory_objects.memory_owner_id
        FROM memory_objects
        WHERE memory_objects.id = contract_dimensions_current.source_memory_id
          AND memory_objects.user_id = contract_dimensions_current.user_id
    ),
    source_mind_id = (
        SELECT memory_objects.source_mind_id
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
        dimension_name
    );

CREATE INDEX IF NOT EXISTS conversations_user_active_mind_idx
    ON conversations(user_id, active_mind_id, status, updated_at DESC)
    WHERE active_mind_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS messages_conversation_active_mind_idx
    ON messages(conversation_id, active_mind_id, seq ASC)
    WHERE active_mind_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS memory_objects_user_owner_idx
    ON memory_objects(user_id, memory_owner_id, status, updated_at DESC, id ASC)
    WHERE memory_owner_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS artifacts_user_owner_idx
    ON artifacts(user_id, memory_owner_id, status, updated_at DESC, id ASC)
    WHERE memory_owner_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS verbatim_pins_user_owner_status_idx
    ON verbatim_pins(user_id, memory_owner_id, status, updated_at DESC, id ASC)
    WHERE memory_owner_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS contract_dimensions_current_owner_idx
    ON contract_dimensions_current(user_id, memory_owner_id, scope_canonical, updated_at DESC)
    WHERE memory_owner_id IS NOT NULL;
