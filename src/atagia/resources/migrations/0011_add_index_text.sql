ALTER TABLE memory_objects ADD COLUMN index_text TEXT;

DROP TRIGGER IF EXISTS memory_objects_fts_ai;
DROP TRIGGER IF EXISTS memory_objects_fts_ad;
DROP TRIGGER IF EXISTS memory_objects_fts_au;
DROP TABLE IF EXISTS memory_objects_fts;

CREATE VIRTUAL TABLE memory_objects_fts
USING fts5(
    canonical_text,
    index_text,
    content='memory_objects',
    content_rowid='_rowid',
    tokenize='unicode61'
);

CREATE TRIGGER memory_objects_fts_ai
AFTER INSERT ON memory_objects
BEGIN
    INSERT INTO memory_objects_fts(rowid, canonical_text, index_text)
    VALUES (new._rowid, new.canonical_text, new.index_text);
END;

CREATE TRIGGER memory_objects_fts_ad
AFTER DELETE ON memory_objects
BEGIN
    INSERT INTO memory_objects_fts(memory_objects_fts, rowid, canonical_text, index_text)
    VALUES ('delete', old._rowid, old.canonical_text, old.index_text);
END;

CREATE TRIGGER memory_objects_fts_au
AFTER UPDATE ON memory_objects
BEGIN
    INSERT INTO memory_objects_fts(memory_objects_fts, rowid, canonical_text, index_text)
    VALUES ('delete', old._rowid, old.canonical_text, old.index_text);
    INSERT INTO memory_objects_fts(rowid, canonical_text, index_text)
    VALUES (new._rowid, new.canonical_text, new.index_text);
END;

INSERT INTO memory_objects_fts(memory_objects_fts) VALUES('rebuild');
