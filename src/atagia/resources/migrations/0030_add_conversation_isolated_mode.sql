ALTER TABLE conversations ADD COLUMN isolated_mode INTEGER NOT NULL DEFAULT 0 CHECK (isolated_mode IN (0, 1));

CREATE INDEX IF NOT EXISTS idx_conversations_user_isolated
    ON conversations(user_id, isolated_mode, updated_at DESC, id ASC);
