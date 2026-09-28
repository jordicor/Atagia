-- The document-frequency cache changes FTS query cleanup, so a stale entry can
-- change retrieval ranking. A wall-clock TTL alone is not an invalidation
-- authority: even in a large corpus, one write can move a token across the
-- inclusive 0.5 filtering boundary. Keep a dedicated SQLite-owned revision
-- that changes only when the corpus scanned by TokenDocumentFrequencyCache
-- changes. Readers pair it with lifecycle_epoch so resetting the counter in a
-- new lifecycle cannot create an ABA cache identity.

ALTER TABLE user_lifecycles
    ADD COLUMN memory_corpus_revision INTEGER NOT NULL DEFAULT 0
    CHECK (memory_corpus_revision >= 0);

CREATE TRIGGER memory_corpus_revision_memory_objects_ai
AFTER INSERT ON memory_objects
BEGIN
    UPDATE user_lifecycles
    SET memory_corpus_revision = memory_corpus_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = NEW.user_id;
END;

-- The current DF scan reads canonical_text and index_text from every row for a
-- user. Status, archival, and other metadata therefore do not change this
-- aggregate and must not invalidate it. If the scan gains another predicate or
-- source column, extend this trigger in the same changeset.
CREATE TRIGGER memory_corpus_revision_memory_objects_au
AFTER UPDATE OF canonical_text, index_text, user_id ON memory_objects
BEGIN
    UPDATE user_lifecycles
    SET memory_corpus_revision = memory_corpus_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id IN (OLD.user_id, NEW.user_id);
END;

CREATE TRIGGER memory_corpus_revision_memory_objects_bd
BEFORE DELETE ON memory_objects
BEGIN
    UPDATE user_lifecycles
    SET memory_corpus_revision = memory_corpus_revision + 1,
        updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
    WHERE user_id = OLD.user_id;
END;
