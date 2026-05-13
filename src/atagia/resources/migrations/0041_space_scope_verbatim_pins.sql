ALTER TABLE verbatim_pins ADD COLUMN space_id TEXT;

ALTER TABLE verbatim_pins ADD COLUMN space_boundary_mode TEXT CHECK (
    space_boundary_mode IN ('focus', 'severance', 'privacy_vault', 'tagged')
    OR space_boundary_mode IS NULL
);

UPDATE verbatim_pins
SET
    space_id = (
        SELECT memory_objects.space_id
        FROM memory_objects
        WHERE memory_objects.id = verbatim_pins.target_id
          AND memory_objects.user_id = verbatim_pins.user_id
    ),
    space_boundary_mode = (
        SELECT memory_objects.space_boundary_mode
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
        AND memory_objects.space_id IS NOT NULL
  );

UPDATE verbatim_pins
SET
    space_id = (
        SELECT messages.space_id
        FROM messages
        JOIN conversations ON conversations.id = messages.conversation_id
        WHERE messages.id = verbatim_pins.target_id
          AND conversations.user_id = verbatim_pins.user_id
    ),
    space_boundary_mode = (
        SELECT spaces.boundary_mode
        FROM messages
        JOIN conversations ON conversations.id = messages.conversation_id
        JOIN spaces
          ON spaces.owner_user_id = conversations.user_id
         AND spaces.id = messages.space_id
        WHERE messages.id = verbatim_pins.target_id
          AND conversations.user_id = verbatim_pins.user_id
    )
WHERE target_kind IN ('message', 'text_span')
  AND space_id IS NULL
  AND EXISTS (
      SELECT 1
      FROM messages
      JOIN conversations ON conversations.id = messages.conversation_id
      WHERE messages.id = verbatim_pins.target_id
        AND conversations.user_id = verbatim_pins.user_id
        AND messages.space_id IS NOT NULL
  );

UPDATE verbatim_pins
SET
    space_id = (
        SELECT conversations.active_space_id
        FROM conversations
        WHERE conversations.id = verbatim_pins.conversation_id
          AND conversations.user_id = verbatim_pins.user_id
    ),
    space_boundary_mode = (
        SELECT spaces.boundary_mode
        FROM conversations
        JOIN spaces
          ON spaces.owner_user_id = conversations.user_id
         AND spaces.id = conversations.active_space_id
        WHERE conversations.id = verbatim_pins.conversation_id
          AND conversations.user_id = verbatim_pins.user_id
    )
WHERE space_id IS NULL
  AND conversation_id IS NOT NULL
  AND EXISTS (
      SELECT 1
      FROM conversations
      WHERE conversations.id = verbatim_pins.conversation_id
        AND conversations.user_id = verbatim_pins.user_id
        AND conversations.active_space_id IS NOT NULL
  );

CREATE INDEX IF NOT EXISTS verbatim_pins_user_space_status_idx
    ON verbatim_pins(user_id, space_id, status, updated_at DESC, id ASC)
    WHERE space_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS verbatim_pins_user_space_boundary_idx
    ON verbatim_pins(user_id, space_boundary_mode, status, updated_at DESC, id ASC)
    WHERE space_id IS NOT NULL;
