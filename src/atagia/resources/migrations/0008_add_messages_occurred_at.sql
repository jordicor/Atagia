ALTER TABLE messages ADD COLUMN occurred_at TEXT;

UPDATE messages
SET occurred_at = created_at
WHERE occurred_at IS NULL;
