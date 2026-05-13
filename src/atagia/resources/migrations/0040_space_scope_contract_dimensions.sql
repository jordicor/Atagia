ALTER TABLE contract_dimensions_current ADD COLUMN space_id TEXT;

ALTER TABLE contract_dimensions_current ADD COLUMN space_boundary_mode TEXT CHECK (
    space_boundary_mode IN ('focus', 'severance', 'privacy_vault', 'tagged')
    OR space_boundary_mode IS NULL
);

ALTER TABLE contract_dimensions_current
    ADD COLUMN space_key TEXT
        GENERATED ALWAYS AS (
            CASE WHEN space_id IS NULL THEN 'n'
                 ELSE 'v:' || length(space_id) || ':' || space_id
            END
        ) VIRTUAL;

UPDATE contract_dimensions_current
SET
    space_id = (
        SELECT memory_objects.space_id
        FROM memory_objects
        WHERE memory_objects.id = contract_dimensions_current.source_memory_id
          AND memory_objects.user_id = contract_dimensions_current.user_id
    ),
    space_boundary_mode = (
        SELECT memory_objects.space_boundary_mode
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
        dimension_name
    );

CREATE INDEX IF NOT EXISTS idx_contract_dimensions_current_space
    ON contract_dimensions_current(user_id, space_id, scope_canonical, updated_at DESC)
    WHERE space_id IS NOT NULL;
