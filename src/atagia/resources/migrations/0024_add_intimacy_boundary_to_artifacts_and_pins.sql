ALTER TABLE artifacts
    ADD COLUMN intimacy_boundary TEXT NOT NULL DEFAULT 'ordinary' CHECK (
        intimacy_boundary IN (
            'ordinary',
            'romantic_private',
            'intimacy_private',
            'intimacy_preference_private',
            'intimacy_boundary',
            'ambiguous_intimate',
            'safety_blocked'
        )
    );

ALTER TABLE artifacts
    ADD COLUMN intimacy_boundary_confidence REAL NOT NULL DEFAULT 0.0 CHECK (
        intimacy_boundary_confidence BETWEEN 0.0 AND 1.0
    );

CREATE INDEX IF NOT EXISTS artifacts_user_status_intimacy_idx
    ON artifacts(user_id, status, intimacy_boundary, updated_at DESC, id ASC);

ALTER TABLE artifact_chunks
    ADD COLUMN intimacy_boundary TEXT NOT NULL DEFAULT 'ordinary' CHECK (
        intimacy_boundary IN (
            'ordinary',
            'romantic_private',
            'intimacy_private',
            'intimacy_preference_private',
            'intimacy_boundary',
            'ambiguous_intimate',
            'safety_blocked'
        )
    );

ALTER TABLE artifact_chunks
    ADD COLUMN intimacy_boundary_confidence REAL NOT NULL DEFAULT 0.0 CHECK (
        intimacy_boundary_confidence BETWEEN 0.0 AND 1.0
    );

CREATE INDEX IF NOT EXISTS artifact_chunks_user_intimacy_idx
    ON artifact_chunks(user_id, intimacy_boundary, updated_at DESC, id ASC);

ALTER TABLE verbatim_pins
    ADD COLUMN intimacy_boundary TEXT NOT NULL DEFAULT 'ordinary' CHECK (
        intimacy_boundary IN (
            'ordinary',
            'romantic_private',
            'intimacy_private',
            'intimacy_preference_private',
            'intimacy_boundary',
            'ambiguous_intimate',
            'safety_blocked'
        )
    );

ALTER TABLE verbatim_pins
    ADD COLUMN intimacy_boundary_confidence REAL NOT NULL DEFAULT 0.0 CHECK (
        intimacy_boundary_confidence BETWEEN 0.0 AND 1.0
    );

CREATE INDEX IF NOT EXISTS verbatim_pins_user_status_intimacy_idx
    ON verbatim_pins(user_id, status, intimacy_boundary, updated_at DESC, id ASC);
