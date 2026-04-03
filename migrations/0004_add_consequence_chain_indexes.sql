CREATE INDEX IF NOT EXISTS idx_consequence_chains_action
    ON consequence_chains(action_memory_id);

CREATE INDEX IF NOT EXISTS idx_consequence_chains_outcome
    ON consequence_chains(outcome_memory_id);

CREATE INDEX IF NOT EXISTS idx_consequence_chains_tendency
    ON consequence_chains(tendency_belief_id);
