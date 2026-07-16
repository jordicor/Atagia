-- Durable selected-transcript replacement and targeted rebuild coordination.

CREATE TABLE transcript_rebuild_workflows (
    id TEXT PRIMARY KEY,
    operation_id TEXT NOT NULL,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    selection_epoch INTEGER NOT NULL CHECK (selection_epoch >= 0),
    transcript_hash TEXT NOT NULL,
    retained_cutoff_message_id TEXT,
    mutation_kind TEXT NOT NULL,
    selected_message_ids_json TEXT NOT NULL,
    abandoned_message_ids_json TEXT NOT NULL,
    supporting_message_ids_json TEXT NOT NULL,
    affected_memory_ids_json TEXT NOT NULL,
    affected_summary_ids_json TEXT NOT NULL,
    orchestrator_job_id TEXT NOT NULL UNIQUE,
    resume_stage TEXT NOT NULL DEFAULT 'preparing' CHECK (
        resume_stage IN (
            'preparing',
            'sources',
            'aggregates',
            'finalizing',
            'ready_to_finalize'
        )
    ),
    embedding_cleanup_completed_at TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0 CHECK (retry_count >= 0),
    stage TEXT NOT NULL CHECK (
        stage IN (
            'preparing',
            'sources',
            'aggregates',
            'finalizing',
            'ready_to_finalize',
            'complete',
            'remediation_required'
        )
    ),
    start_derivation_revision INTEGER NOT NULL CHECK (start_derivation_revision >= 0),
    completion_derivation_revision INTEGER CHECK (completion_derivation_revision >= 0),
    error_code TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    UNIQUE(user_id, conversation_id, operation_id),
    UNIQUE(user_id, conversation_id, selection_epoch)
);

CREATE INDEX idx_transcript_rebuild_workflows_user_stage
    ON transcript_rebuild_workflows(user_id, stage, updated_at, id);
CREATE INDEX idx_transcript_rebuild_workflows_conversation_epoch
    ON transcript_rebuild_workflows(
        user_id,
        conversation_id,
        selection_epoch DESC,
        id
    );

CREATE TABLE conversation_transcript_selections (
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    selection_epoch INTEGER NOT NULL CHECK (selection_epoch >= 0),
    transcript_hash TEXT NOT NULL,
    current_workflow_id TEXT NOT NULL REFERENCES transcript_rebuild_workflows(id),
    state TEXT NOT NULL CHECK (
        state IN ('preparing', 'rebuilding', 'complete', 'remediation_required')
    ),
    updated_at TEXT NOT NULL,
    PRIMARY KEY(user_id, conversation_id),
    UNIQUE(current_workflow_id)
);

CREATE INDEX idx_conversation_transcript_selections_user_state
    ON conversation_transcript_selections(user_id, state, updated_at, conversation_id);

CREATE TABLE transcript_rebuild_targets (
    workflow_id TEXT NOT NULL REFERENCES transcript_rebuild_workflows(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    message_id TEXT NOT NULL,
    source_kind TEXT NOT NULL CHECK (
        source_kind IN ('selected', 'supporting', 'interrupted_job')
    ),
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    require_contract INTEGER NOT NULL DEFAULT 0 CHECK (require_contract IN (0, 1)),
    PRIMARY KEY(workflow_id, message_id)
);

CREATE INDEX idx_transcript_rebuild_targets_workflow_kind
    ON transcript_rebuild_targets(workflow_id, source_kind, message_id);

ALTER TABLE worker_job_runs
    ADD COLUMN transcript_rebuild_id TEXT
    REFERENCES transcript_rebuild_workflows(id) ON DELETE SET NULL;

CREATE INDEX idx_worker_job_runs_transcript_rebuild
    ON worker_job_runs(transcript_rebuild_id, status, job_type, queued_at, job_id);

-- Normal message writes for this user are held while shared derived state is
-- being rebuilt. The replacement transaction itself mutates messages while the
-- selection row is still in the uncommitted preparing state.
CREATE TRIGGER transcript_rebuild_block_message_insert
BEFORE INSERT ON messages
WHEN EXISTS (
    SELECT 1
    FROM conversations AS c
    JOIN conversation_transcript_selections AS selection
      ON selection.user_id = c.user_id
    WHERE c.id = new.conversation_id
      AND selection.state IN ('rebuilding', 'remediation_required')
)
BEGIN
    SELECT RAISE(ABORT, 'selected transcript rebuild blocks user message writes');
END;

CREATE TRIGGER transcript_rebuild_block_message_update
BEFORE UPDATE ON messages
WHEN EXISTS (
    SELECT 1
    FROM conversations AS c
    JOIN conversation_transcript_selections AS selection
      ON selection.user_id = c.user_id
    WHERE c.id = old.conversation_id
      AND selection.state IN ('rebuilding', 'remediation_required')
)
BEGIN
    SELECT RAISE(ABORT, 'selected transcript rebuild blocks user message writes');
END;

CREATE TRIGGER transcript_rebuild_block_message_delete
BEFORE DELETE ON messages
WHEN EXISTS (
    SELECT 1
    FROM conversations AS c
    JOIN conversation_transcript_selections AS selection
      ON selection.user_id = c.user_id
    WHERE c.id = old.conversation_id
      AND selection.state IN ('rebuilding', 'remediation_required')
)
BEGIN
    SELECT RAISE(ABORT, 'selected transcript rebuild blocks user message writes');
END;
