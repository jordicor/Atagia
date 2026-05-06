ALTER TABLE graph_entity_mentions ADD COLUMN source_occurrence_key TEXT NOT NULL DEFAULT '';

ALTER TABLE graph_relationship_sources ADD COLUMN source_occurrence_key TEXT NOT NULL DEFAULT '';

DROP INDEX IF EXISTS graph_relationship_sources_unique_idx;

CREATE UNIQUE INDEX IF NOT EXISTS graph_relationship_sources_unique_idx
    ON graph_relationship_sources(
        user_id,
        relationship_id,
        source_kind,
        source_id,
        source_occurrence_key,
        quote_hash
    );
