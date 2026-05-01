ALTER TABLE memory_objects
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

ALTER TABLE memory_objects
    ADD COLUMN intimacy_boundary_confidence REAL NOT NULL DEFAULT 0.0 CHECK (
        intimacy_boundary_confidence BETWEEN 0.0 AND 1.0
    );

CREATE INDEX IF NOT EXISTS idx_mo_user_status_intimacy
    ON memory_objects(user_id, status, intimacy_boundary);

ALTER TABLE summary_views
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

ALTER TABLE summary_views
    ADD COLUMN intimacy_boundary_confidence REAL NOT NULL DEFAULT 0.0 CHECK (
        intimacy_boundary_confidence BETWEEN 0.0 AND 1.0
    );

CREATE INDEX IF NOT EXISTS idx_sv_user_kind_intimacy
    ON summary_views(user_id, summary_kind, intimacy_boundary, created_at);

ALTER TABLE conversation_topics
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

ALTER TABLE conversation_topics
    ADD COLUMN intimacy_boundary_confidence REAL NOT NULL DEFAULT 0.0 CHECK (
        intimacy_boundary_confidence BETWEEN 0.0 AND 1.0
    );

CREATE INDEX IF NOT EXISTS conversation_topics_user_conversation_intimacy_idx
    ON conversation_topics(user_id, conversation_id, status, intimacy_boundary, last_touched_at DESC, id ASC);
