-- Per-turn LLM telemetry gains latency per purpose (CS-1.5). Until now the
-- breakdown answered "how many calls did each stage make" but not "where did
-- the wall time go", which is the half a latency workstream actually needs.
--
-- ONE column, not two. Calls and latency are keyed by the same open set of
-- purpose labels, so two parallel maps would drift the moment one gained a
-- label the other did not have. The value is therefore an object of objects:
--   {"need_detection": {"calls": 3, "latency_ms": 412.5}, ...}
-- The CHECK below enforces that shape as far as SQLite can see it (valid JSON,
-- object at the top level); TurnTelemetry.__post_init__ enforces the rest --
-- the per-purpose call counts sum to llm_total_calls and the per-purpose
-- latency never exceeds llm_total_latency_ms.
--
-- The 0069 counts are CARRIED FORWARD, then the old column is dropped. Rows
-- written under 0069 recorded real calls per purpose but never measured latency
-- per purpose, so each carried entry reads
--   {"<purpose>": {"calls": n, "latency_ms": null}}
-- The count survives because it is a measurement that was actually taken; the
-- latency is explicitly absent. That null is 0069's own "written before this
-- migration" convention applied one level down, NOT the fabricated 0.0 that
-- 0069 refused for its own columns. Being unable to carry the latency honestly
-- was never a reason to throw the counts away with it.
--
-- Three states are therefore distinguishable on this column alone:
--   NULL                          -- no breakdown was ever recorded (pre-0069).
--   {"p": {"calls": n, "latency_ms": null}}  -- counted under 0069, never timed.
--   {"p": {"calls": n, "latency_ms": x}}     -- written by the current engine.
-- A null latency_ms is an ABSENT measurement and must never be read as zero.
-- Writers always supply a real number: TurnTelemetry.__post_init__ rejects a
-- null on the write path, so a null can only ever be this backfill.
--
-- SQLITE FLOOR: 3.35.0. The final statement is this repo's first ALTER TABLE
-- ... DROP COLUMN; every earlier column removal went through a table rebuild,
-- which has no version floor. On an older SQLite the DROP raises, and because
-- the migration runner keeps one migration's statements in a single
-- transaction, the whole migration rolls back and the schema stays on 0069 --
-- a hard, visible failure rather than a half-applied schema. The floor is
-- stated rather than guarded: a rebuild of retrieval_events would cost a full
-- table copy on every deployment to accommodate a SQLite older than any
-- environment that can run this package's Python floor.

ALTER TABLE retrieval_events ADD COLUMN llm_by_purpose_json TEXT
    CHECK (
        llm_by_purpose_json IS NULL
        OR (
            json_valid(llm_by_purpose_json)
            AND json_type(llm_by_purpose_json) = 'object'
        )
    );

-- Carry the counts across before the column holding them disappears. Scoped to
-- rows that actually have a breakdown: json_each over NULL yields no rows and
-- json_group_object over no rows returns '{}', which would turn "never
-- recorded" into "recorded an empty breakdown".
UPDATE retrieval_events
SET llm_by_purpose_json = (
        SELECT json_group_object(
            purpose_entry.key,
            json_object('calls', purpose_entry.value, 'latency_ms', NULL)
        )
        FROM json_each(retrieval_events.llm_calls_by_purpose_json) AS purpose_entry
    )
WHERE llm_calls_by_purpose_json IS NOT NULL;

ALTER TABLE retrieval_events DROP COLUMN llm_calls_by_purpose_json;
