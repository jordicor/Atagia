ALTER TABLE memory_objects ADD COLUMN tension_score REAL NOT NULL DEFAULT 0.0;
ALTER TABLE memory_objects ADD COLUMN tension_updated_at TEXT;

CREATE INDEX IF NOT EXISTS idx_mo_tension
    ON memory_objects(user_id, tension_score)
    WHERE tension_score > 0 AND object_type = 'belief';
