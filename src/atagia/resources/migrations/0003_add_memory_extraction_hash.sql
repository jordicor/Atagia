ALTER TABLE memory_objects
    ADD COLUMN extraction_hash TEXT;

WITH ranked_hashes AS (
    SELECT
        _rowid,
        user_id,
        json_extract(payload_json, '$.extraction_hash') AS extracted_hash,
        ROW_NUMBER() OVER (
            PARTITION BY user_id, json_extract(payload_json, '$.extraction_hash')
            ORDER BY created_at ASC, id ASC
        ) AS ordinal
    FROM memory_objects
    WHERE json_extract(payload_json, '$.extraction_hash') IS NOT NULL
)
UPDATE memory_objects
SET extraction_hash = (
    SELECT extracted_hash
    FROM ranked_hashes
    WHERE ranked_hashes._rowid = memory_objects._rowid
)
WHERE _rowid IN (
    SELECT _rowid
    FROM ranked_hashes
    WHERE ordinal = 1
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_memory_objects_user_extraction_hash
    ON memory_objects(user_id, extraction_hash)
    WHERE extraction_hash IS NOT NULL;
