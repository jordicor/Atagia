ALTER TABLE users
    ADD COLUMN memory_privacy_mode TEXT NOT NULL DEFAULT 'balanced'
        CHECK (memory_privacy_mode IN ('balanced', 'trusted_private'));

CREATE INDEX IF NOT EXISTS idx_users_memory_privacy_mode
    ON users(id, memory_privacy_mode);
