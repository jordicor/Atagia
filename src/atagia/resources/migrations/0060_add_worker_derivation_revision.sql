-- Durable revision fence for background work derived from canonical user state.

ALTER TABLE user_lifecycles
    ADD COLUMN derivation_revision INTEGER NOT NULL DEFAULT 0
    CHECK (derivation_revision >= 0);

ALTER TABLE worker_job_runs
    ADD COLUMN derivation_revision INTEGER NOT NULL DEFAULT 0
    CHECK (derivation_revision >= 0);
