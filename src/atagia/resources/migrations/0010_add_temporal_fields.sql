ALTER TABLE memory_objects ADD COLUMN temporal_type TEXT NOT NULL DEFAULT 'unknown';

CREATE INDEX IF NOT EXISTS idx_mo_temporal_expiry
    ON memory_objects(valid_to, status, object_type)
    WHERE valid_to IS NOT NULL AND status = 'active';
