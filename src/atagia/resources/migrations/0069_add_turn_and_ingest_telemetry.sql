-- Per-turn latency and LLM-call telemetry become queryable columns instead of
-- living only inside outcome_json, and every memory records when its source
-- message arrived so ingest freshness can be measured without guessing.
--
-- The scalars below are the ones latency work charts (turn wall time, retrieval
-- wall time, provider round-trips, provider latency), so they are typed columns.
-- The two breakdown maps stay JSON because their keys are open sets.
--
-- `turn_duration_ms` below was misnamed: it is stamped when the row is written,
-- not when the turn ends. Migration 0072 renames it to
-- `turn_to_event_write_wall_ms` and records the measured tail it excludes.
--
-- All measurement columns are nullable: NULL means "written before this
-- migration", which is honest, while 0 would be a fabricated measurement. Every
-- row written from now on supplies them (RetrievalEventRepository requires the
-- telemetry argument).

ALTER TABLE retrieval_events ADD COLUMN turn_surface TEXT NOT NULL DEFAULT 'chat'
    CHECK (turn_surface IN ('chat', 'context', 'proxy_completion', 'proxy_stream'));

ALTER TABLE retrieval_events ADD COLUMN turn_duration_ms REAL
    CHECK (turn_duration_ms IS NULL OR turn_duration_ms >= 0.0);

ALTER TABLE retrieval_events ADD COLUMN retrieval_duration_ms REAL
    CHECK (retrieval_duration_ms IS NULL OR retrieval_duration_ms >= 0.0);

ALTER TABLE retrieval_events ADD COLUMN llm_total_calls INTEGER
    CHECK (llm_total_calls IS NULL OR llm_total_calls >= 0);

ALTER TABLE retrieval_events ADD COLUMN llm_failed_calls INTEGER
    CHECK (llm_failed_calls IS NULL OR llm_failed_calls >= 0);

ALTER TABLE retrieval_events ADD COLUMN llm_total_latency_ms REAL
    CHECK (llm_total_latency_ms IS NULL OR llm_total_latency_ms >= 0.0);

ALTER TABLE retrieval_events ADD COLUMN llm_calls_by_purpose_json TEXT
    CHECK (
        llm_calls_by_purpose_json IS NULL
        OR (
            json_valid(llm_calls_by_purpose_json)
            AND json_type(llm_calls_by_purpose_json) = 'object'
        )
    );

ALTER TABLE retrieval_events ADD COLUMN stage_timings_ms_json TEXT
    CHECK (
        stage_timings_ms_json IS NULL
        OR (
            json_valid(stage_timings_ms_json)
            AND json_type(stage_timings_ms_json) = 'object'
        )
    );

-- Aggregation is always per user and almost always per surface over a time
-- window ("what did chat turns cost this week vs proxy turns").
CREATE INDEX idx_retrieval_events_turn_surface
    ON retrieval_events(user_id, turn_surface, created_at);

-- get_context is retry-safe: re-posting the same message_id returns the same
-- message row instead of creating a second one. Its retrieval event has to
-- inherit that property or every aggregate over this ledger over-counts on
-- exactly the path the API makes retry-safe, so the retry overwrites the
-- existing row rather than minting a new id. This partial unique index makes
-- that an enforced invariant instead of a convention the writer must remember.
-- It is deliberately scoped to the 'context' surface only: chat writes one row
-- per turn keyed by its own request message, and the proxy surfaces reuse the
-- same request_message_id across a retried turn while carrying a different
-- turn_surface value, so a full-table unique index would reject legitimate rows.
CREATE UNIQUE INDEX uq_retrieval_events_context_request
    ON retrieval_events(user_id, conversation_id, request_message_id)
    WHERE turn_surface = 'context';

-- Ingest freshness, arrival half. This column records when the message that
-- produced the memory arrived (the server-side messages.created_at, not the
-- caller-supplied occurred_at, which can be backdated to any historical
-- instant). Ingest latency is
--   queryable_at - source_message_created_at
-- computed from two stored timestamps rather than from a delta nobody could
-- re-derive.
--
-- The right-hand side was originally created_at, on the argument that
-- memory_objects_fts is populated by an AFTER INSERT trigger and so a memory
-- enters the searchable corpus in the same statement that creates it. That was
-- only ever true for rows born status = 'active': being in the FTS index is not
-- the same as being retrievable, retrieval filters candidates on
-- status = 'active', and extraction routinely mints rows that are not. 0071
-- removed the guess by adding memory_objects.queryable_at, stamped at the first
-- instant a row actually reached 'active' -- see that migration for where it is
-- written, why it stays NULL, and how an aggregate scopes itself. An
-- ingest-latency aggregate no longer needs a status filter to be honest; it
-- filters on queryable_at IS NOT NULL, plus object_type/source_kind per below.
--
-- Only memories minted from exactly one arriving message carry the stamp, and
-- exactly three producers qualify:
--   * MemoryExtractor._persist_result (memory/extractor.py) -- evidence and
--     beliefs extracted from the message the ingest job is processing.
--   * ContractProjector.project (memory/contract_projection.py) -- interaction
--     contract signals grounded in that same message.
--   * ConsequenceChainBuilder._create_outcome_memory (memory/consequence_builder.py)
--     -- the outcome half of a consequence chain, which is the only part
--     grounded in the current message; build_chain resolves the stamp and hands
--     it down.
-- All three read created_at off a row fetched with
-- MessageRepository.get_message(id, user_id), so the stamp is always a
-- server-side arrival instant for a message that belongs to the requesting user
-- and to the active conversation.
--
-- The remaining producers leave it NULL because they have no single arriving
-- message to measure against, not because they were overlooked:
--   * ConsequenceChainBuilder._create_inferred_action_memory -- reconstructs an
--     assistant action after the fact; its source message is an earlier
--     assistant turn when the detector linked one and the current turn
--     otherwise, so the same column would mean two different intervals
--     depending on an LLM link decision.
--   * ConsequenceChainBuilder._create_tendency_memory -- inferred from the
--     action and the later outcome together, spanning two messages.
--   * BeliefReviser._create_belief_memory (memory/belief_reviser.py) --
--     successor and exception beliefs merge a base belief with every new
--     evidence row that triggered the revision.
--   * RevisionWorker._create_promoted_belief (workers/revision_worker.py) --
--     promotion happens because a claim recurred across distinct conversations
--     and sessions, which is the opposite of a single-message derivation.
--   * MemoryObjectRepository.upsert_summary_mirror (core/repositories.py) --
--     a summary view compacts many source objects by definition.
-- Those rows are indistinguishable from pre-migration rows on this column
-- alone, so ingest-latency aggregates must filter on object_type/source_kind
-- (and on queryable_at, per above) rather than assume every NULL is a backfill
-- gap.
ALTER TABLE memory_objects ADD COLUMN source_message_created_at TEXT;
