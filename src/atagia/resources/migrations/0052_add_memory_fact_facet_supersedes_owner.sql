ALTER TABLE memory_fact_facets ADD COLUMN updated_at TEXT;

UPDATE memory_fact_facets
SET updated_at = created_at
WHERE updated_at IS NULL;

CREATE TRIGGER IF NOT EXISTS memory_fact_facets_supersedes_owner_bi
BEFORE INSERT ON memory_fact_facets
WHEN new.supersedes_fact_id IS NOT NULL
BEGIN
    SELECT RAISE(ABORT, 'memory_fact_facets supersedes_fact_id must match user_id')
    WHERE NOT EXISTS (
        SELECT 1
        FROM memory_fact_facets AS superseded
        WHERE superseded.id = new.supersedes_fact_id
          AND superseded.user_id = new.user_id
    );
END;

CREATE TRIGGER IF NOT EXISTS memory_fact_facets_supersedes_owner_bu
BEFORE UPDATE OF user_id, supersedes_fact_id ON memory_fact_facets
WHEN new.supersedes_fact_id IS NOT NULL
BEGIN
    SELECT RAISE(ABORT, 'memory_fact_facets supersedes_fact_id must match user_id')
    WHERE NOT EXISTS (
        SELECT 1
        FROM memory_fact_facets AS superseded
        WHERE superseded.id = new.supersedes_fact_id
          AND superseded.user_id = new.user_id
    );
END;
