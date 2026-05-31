CREATE TABLE IF NOT EXISTS memory_retrieval_surfaces (
    _rowid INTEGER PRIMARY KEY,
    id TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    memory_id TEXT NOT NULL REFERENCES memory_objects(id) ON DELETE CASCADE,
    surface_type TEXT NOT NULL CHECK (
        surface_type IN ('pivot', 'anchor', 'alias', 'corpus_surface')
    ),
    anchor_type TEXT CHECK (
        anchor_type IN (
            'proper_name',
            'person',
            'organization',
            'location',
            'code',
            'quantity',
            'date_time',
            'address',
            'quoted_phrase',
            'attribute',
            'concept',
            'unknown'
        )
        OR anchor_type IS NULL
    ),
    alias_kind TEXT CHECK (
        alias_kind IN (
            'translation',
            'transliteration',
            'spelling_variant',
            'acronym_expansion',
            'domain_synonym',
            'corpus_surface'
        )
        OR alias_kind IS NULL
    ),
    language_code TEXT,
    surface_text TEXT NOT NULL CHECK (TRIM(surface_text) <> ''),
    surface_key TEXT NOT NULL CHECK (TRIM(surface_key) <> ''),
    preserve_verbatim INTEGER NOT NULL DEFAULT 0 CHECK (preserve_verbatim IN (0, 1)),
    non_evidential INTEGER NOT NULL DEFAULT 1 CHECK (non_evidential = 1),
    confidence REAL NOT NULL DEFAULT 0.5 CHECK (confidence BETWEEN 0.0 AND 1.0),
    derivation_kind TEXT NOT NULL CHECK (TRIM(derivation_kind) <> ''),
    derivation_model TEXT,
    derivation_prompt_version TEXT,
    derivation_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'stale', 'deleted')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    language_key TEXT GENERATED ALWAYS AS (
        CASE WHEN language_code IS NULL THEN 'n'
             ELSE 'v:' || length(language_code) || ':' || language_code
        END
    ) VIRTUAL,
    anchor_type_key TEXT GENERATED ALWAYS AS (
        CASE WHEN anchor_type IS NULL THEN 'n'
             ELSE 'v:' || length(anchor_type) || ':' || anchor_type
        END
    ) VIRTUAL,
    alias_kind_key TEXT GENERATED ALWAYS AS (
        CASE WHEN alias_kind IS NULL THEN 'n'
             ELSE 'v:' || length(alias_kind) || ':' || alias_kind
        END
    ) VIRTUAL
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_memory_retrieval_surfaces_signature
    ON memory_retrieval_surfaces(
        user_id,
        memory_id,
        surface_type,
        language_key,
        anchor_type_key,
        alias_kind_key,
        surface_key
    );

CREATE INDEX IF NOT EXISTS idx_memory_retrieval_surfaces_user_memory
    ON memory_retrieval_surfaces(user_id, memory_id, status, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_memory_retrieval_surfaces_user_type
    ON memory_retrieval_surfaces(user_id, surface_type, status, updated_at DESC);

CREATE TRIGGER IF NOT EXISTS memory_retrieval_surfaces_owner_bi
BEFORE INSERT ON memory_retrieval_surfaces
BEGIN
    SELECT RAISE(ABORT, 'memory_retrieval_surfaces user_id must match memory_objects.user_id')
    WHERE NOT EXISTS (
        SELECT 1
        FROM memory_objects
        WHERE memory_objects.id = new.memory_id
          AND memory_objects.user_id = new.user_id
    );
END;

CREATE TRIGGER IF NOT EXISTS memory_retrieval_surfaces_owner_bu
BEFORE UPDATE OF user_id, memory_id ON memory_retrieval_surfaces
BEGIN
    SELECT RAISE(ABORT, 'memory_retrieval_surfaces user_id must match memory_objects.user_id')
    WHERE NOT EXISTS (
        SELECT 1
        FROM memory_objects
        WHERE memory_objects.id = new.memory_id
          AND memory_objects.user_id = new.user_id
    );
END;

CREATE VIRTUAL TABLE IF NOT EXISTS memory_retrieval_surfaces_fts
USING fts5(
    surface_text,
    content='memory_retrieval_surfaces',
    content_rowid='_rowid',
    tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS memory_retrieval_surfaces_fts_ai
AFTER INSERT ON memory_retrieval_surfaces
BEGIN
    INSERT INTO memory_retrieval_surfaces_fts(rowid, surface_text)
    VALUES (new._rowid, new.surface_text);
END;

CREATE TRIGGER IF NOT EXISTS memory_retrieval_surfaces_fts_ad
AFTER DELETE ON memory_retrieval_surfaces
BEGIN
    INSERT INTO memory_retrieval_surfaces_fts(memory_retrieval_surfaces_fts, rowid, surface_text)
    VALUES ('delete', old._rowid, old.surface_text);
END;

CREATE TRIGGER IF NOT EXISTS memory_retrieval_surfaces_fts_au
AFTER UPDATE ON memory_retrieval_surfaces
BEGIN
    INSERT INTO memory_retrieval_surfaces_fts(memory_retrieval_surfaces_fts, rowid, surface_text)
    VALUES ('delete', old._rowid, old.surface_text);
    INSERT INTO memory_retrieval_surfaces_fts(rowid, surface_text)
    VALUES (new._rowid, new.surface_text);
END;
