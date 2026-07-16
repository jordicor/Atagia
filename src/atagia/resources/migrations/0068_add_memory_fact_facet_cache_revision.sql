-- Fact/facet retrieval rows invalidate context caches without changing ICP sources.

CREATE TRIGGER cache_source_memory_fact_facets_ai
AFTER INSERT ON memory_fact_facets
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = NEW.user_id;
END;

CREATE TRIGGER cache_source_memory_fact_facets_au
AFTER UPDATE ON memory_fact_facets
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id IN (OLD.user_id, NEW.user_id);
END;

CREATE TRIGGER cache_source_memory_fact_facets_bd
BEFORE DELETE ON memory_fact_facets
BEGIN
    UPDATE user_lifecycles
    SET cache_revision = cache_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = OLD.user_id;
END;
