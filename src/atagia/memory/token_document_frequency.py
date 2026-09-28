"""Per-user document-frequency statistics used to prune FTS query tokens.

A token that appears in most of a user's memories carries no lexical signal:
matching on it narrows nothing. ``CandidateSearch`` drops such tokens from AND
queries, and this module supplies the statistics that decide which tokens those
are.

Why revision-keyed invalidation and a TTL
-----------------------------------------
The cache changes which FTS terms survive query cleanup, so freshness is a
ranking invariant rather than a best-effort performance choice. A corpus of any
size can sit exactly on the inclusive filtering boundary; one insert can move a
ratio from 0.5 to just below it. Migration 0073 therefore adds the dedicated
SQLite-owned ``user_lifecycles.memory_corpus_revision`` and triggers it on every
change to the rows or text this module scans. Every cache read verifies both the
revision and the immutable lifecycle epoch before reuse, including across
processes and lifecycle retirement/recreation.

The TTL remains a bounded-lifetime defence, not the invalidation authority. It
applies only above ``_TTL_MIN_CORPUS_DOCUMENTS``; below that size the scan is
cheap enough to repeat. Erasure also calls ``forget`` so data derived from the
erased corpus leaves the erasing runtime's process memory even when canonical
cleanup finishes before transient cleanup. Other runtimes reject that entry on
their next read through the SQLite identity check.

The cache lives for the process lifetime, one instance per database, and is
owned by ``AppRuntime``. It must be injected into every consumer rather than
default-constructed, because a cache allocated per request is indistinguishable
from no cache at all.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType

import aiosqlite

from atagia.core.clock import Clock
from atagia.memory.retrieval_planner import FTS_TOKEN_PATTERN

# Ratio at or above which a token is considered corpus-ubiquitous and dropped
# from an AND query. Owned here because the cache prunes at this same value.
TOKEN_DOC_RATIO_MAX_RATIO = 0.5

# Below this many documents the ratios cannot discriminate: a token in 5 of 8
# memories is not evidence of anything.
_MIN_CORPUS_DOCUMENTS = 10

# How long a user's statistics stay usable before being recomputed.
_CACHE_TTL_SECONDS = 300.0

# Corpus size below which the TTL is skipped and the scan repeats every time.
# The SQLite revision already prevents stale reuse at every size; below this
# threshold the scan itself costs only a few milliseconds (about 10
# microseconds per document when measured), so retaining the entry provides no
# material benefit.
_TTL_MIN_CORPUS_DOCUMENTS = 500

# Hard bound on the number of users retained. Entries only hold ubiquitous
# tokens (see ``_ubiquitous_ratios``), so each one is small, but the number of
# users a multi-tenant process sees is not bounded by anything else.
_MAX_CACHED_USERS = 512

_CORPUS_SCAN_SQL = """
    SELECT canonical_text, index_text
    FROM memory_objects
    WHERE user_id = ?
"""

_CORPUS_REVISION_SQL = """
    SELECT lifecycle_epoch, memory_corpus_revision
    FROM user_lifecycles
    WHERE user_id = ?
"""

_EMPTY_RATIOS: Mapping[str, float] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    """One user's statistics, when they were captured, and over how much."""

    captured_at: datetime
    ratios: Mapping[str, float]
    total_documents: int
    corpus_identity: tuple[str, int] | None


def _ubiquitous_ratios(
    document_frequencies: dict[str, int],
    total_documents: int,
) -> Mapping[str, float]:
    """Keep only the tokens a query would actually be filtered on.

    A token below ``TOKEN_DOC_RATIO_MAX_RATIO`` is treated as informative, and
    a token absent from the mapping is treated the same way, so dropping the
    below-threshold entries preserves the outcome for every consumer that
    filters at this threshold. It is also what keeps an entry small: at most
    ``2 * average distinct tokens per document`` tokens can sit above a 0.5
    ratio, whereas the full vocabulary grows with the corpus.

    The result is a read-only view because it is handed to every caller by
    reference; a consumer mutating it would corrupt later turns.
    """
    if total_documents < _MIN_CORPUS_DOCUMENTS:
        return _EMPTY_RATIOS
    return MappingProxyType(
        {
            token: frequency / total_documents
            for token, frequency in document_frequencies.items()
            if frequency / total_documents >= TOKEN_DOC_RATIO_MAX_RATIO
        }
    )


class TokenDocumentFrequencyCache:
    """Process-lifetime cache of per-user token document-frequency ratios.

    One instance per database. Concurrent first turns for the same user may
    each run the scan. Duplicate work is accepted rather than serialized behind
    a lock; every reused result must still match SQLite's lifecycle-scoped
    corpus identity.
    """

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()

    async def ubiquitous_token_ratios(
        self,
        connection: aiosqlite.Connection,
        clock: Clock,
        user_id: str,
    ) -> Mapping[str, float]:
        """Return ratios for this user's corpus-ubiquitous tokens.

        An empty mapping means no token is ubiquitous, which includes the case
        of a corpus too small to measure. Both mean "drop nothing".
        """
        now = clock.now()
        corpus_identity = await self._current_corpus_identity(connection, user_id)
        cached = self._entries.get(user_id)
        if cached is not None and self._is_usable(
            cached,
            now,
            corpus_identity=corpus_identity,
        ):
            self._entries.move_to_end(user_id)
            return cached.ratios
        # A different process can commit an edit or erasure while this process
        # scans. Verify the identity after the scan before publishing it into
        # process memory; retry once, then fail open to "drop no FTS tokens"
        # rather than retain a result captured across a moving corpus.
        for _attempt in range(2):
            entry = await self._scan_user_corpus(
                connection,
                user_id,
                captured_at=now,
                corpus_identity=corpus_identity,
            )
            verified_identity = await self._current_corpus_identity(
                connection,
                user_id,
            )
            if verified_identity == corpus_identity:
                self._entries[user_id] = entry
                # Re-assigning an existing key keeps its old position, so
                # refreshing an expired entry must move it explicitly.
                self._entries.move_to_end(user_id)
                while len(self._entries) > _MAX_CACHED_USERS:
                    self._entries.popitem(last=False)
                return entry.ratios
            self._entries.pop(user_id, None)
            corpus_identity = verified_identity
        return _EMPTY_RATIOS

    def forget(self, user_id: str) -> None:
        """Drop a user's statistics immediately.

        Revision and epoch checks already prevent reuse after erasure. Explicit
        eviction additionally removes statistics derived from erased content
        from this process's memory instead of retaining them until LRU eviction.
        """
        self._entries.pop(user_id, None)

    @staticmethod
    def _is_usable(
        entry: _CacheEntry,
        now: datetime,
        *,
        corpus_identity: tuple[str, int] | None,
    ) -> bool:
        if entry.corpus_identity != corpus_identity:
            return False
        if entry.total_documents < _TTL_MIN_CORPUS_DOCUMENTS:
            return False
        return (now - entry.captured_at).total_seconds() < _CACHE_TTL_SECONDS

    @staticmethod
    async def _current_corpus_identity(
        connection: aiosqlite.Connection,
        user_id: str,
    ) -> tuple[str, int] | None:
        """Read the lifecycle-scoped SQLite authority for this corpus."""
        cursor = await connection.execute(_CORPUS_REVISION_SQL, (user_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return str(row["lifecycle_epoch"]), int(row["memory_corpus_revision"])

    @staticmethod
    async def _scan_user_corpus(
        connection: aiosqlite.Connection,
        user_id: str,
        *,
        captured_at: datetime,
        corpus_identity: tuple[str, int] | None,
    ) -> _CacheEntry:
        """Tokenize the user's indexed text and count documents per token.

        Reads ``memory_objects`` directly. ``memory_objects_fts`` is an
        external-content table over these same columns, and a scan of it
        without a MATCH resolves every row back to ``memory_objects`` through
        the virtual-table layer -- verified by emptying the index, after which
        the old join still returned every row. The join therefore only ever
        added indirection.

        The scan is deliberately unfiltered by status and archival: it covers a
        strictly larger corpus than the one ``CandidateSearch`` searches. That
        divergence shifts every ratio and is tracked separately from this
        module's caching contract.
        """
        cursor = await connection.execute(_CORPUS_SCAN_SQL, (user_id,))
        rows = await cursor.fetchall()
        if rows and corpus_identity is None:
            raise RuntimeError(
                "Memory corpus has no lifecycle-scoped revision authority for "
                f"user_id={user_id}"
            )
        document_frequencies: dict[str, int] = {}
        for row in rows:
            document_tokens = {
                token.lower()
                for value in (row["canonical_text"], row["index_text"])
                for token in FTS_TOKEN_PATTERN.findall(str(value or ""))
            }
            for token in document_tokens:
                document_frequencies[token] = document_frequencies.get(token, 0) + 1
        return _CacheEntry(
            captured_at=captured_at,
            ratios=_ubiquitous_ratios(document_frequencies, len(rows)),
            total_documents=len(rows),
            corpus_identity=corpus_identity,
        )
