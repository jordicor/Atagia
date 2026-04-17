-- Wave 1 batch 2 (1-C): raw evidence channel privacy filtering.
-- Adds an explicit privacy_ceiling column to assistant_modes so the raw
-- message search channel can enforce mode-level privacy at the SQL layer
-- before any ranking happens.

ALTER TABLE assistant_modes
    ADD COLUMN privacy_ceiling INTEGER NOT NULL DEFAULT 0;

-- Index speeds up the raw-message channel join pattern
-- (conversations -> assistant_modes -> privacy_ceiling) and the
-- conversation_id filter used by deduping logic.
CREATE INDEX IF NOT EXISTS idx_conversations_mode_user
    ON conversations(assistant_mode_id, user_id);
