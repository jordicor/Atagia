ALTER TABLE memory_fact_facets
ADD COLUMN surface_class TEXT NOT NULL DEFAULT 'generic'
CHECK (surface_class IN ('structured', 'generic'));

CREATE INDEX IF NOT EXISTS idx_memory_fact_facets_user_surface_class
    ON memory_fact_facets(user_id, surface_class, current_state, created_at DESC, id ASC);
