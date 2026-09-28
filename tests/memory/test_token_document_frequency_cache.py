"""Lifetime and fail-fast contract of the per-user corpus statistics cache.

The defect these tests exist to prevent: the cache used to be allocated inside
``CandidateSearch.__init__``, and every production construction site builds a
new ``CandidateSearch`` per request, so the cache never survived a single turn
and its TTL was unreachable configuration. Every assertion below counts real
corpus-scan queries rather than a cache-hit flag, because a hit flag can report
success while the scan still runs.
"""

from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import aiosqlite
import pytest

from atagia.app import AppRuntime
from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.db_sqlite import initialize_database
from atagia.memory.candidate_search import CandidateSearch
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.memory.retrieval_planner import build_retrieval_fts_queries
from atagia.memory.token_document_frequency import (
    _CACHE_TTL_SECONDS,
    _CORPUS_SCAN_SQL,
    _MAX_CACHED_USERS,
    _TTL_MIN_CORPUS_DOCUMENTS,
    TOKEN_DOC_RATIO_MAX_RATIO,
    TokenDocumentFrequencyCache,
)
from atagia.services.embeddings import NoneBackend
from atagia.services.retrieval_pipeline import RetrievalPipeline
from atagia.models.schemas_memory import (
    MemoryScope,
    MemoryStatus,
    PlannedSubQuery,
    RetrievalPlan,
)

MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "migrations"
)
MANIFESTS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "manifests"
)

# Large enough that the TTL applies. Below ``_TTL_MIN_CORPUS_DOCUMENTS`` the
# cache deliberately rescans because the scan is cheap; SQLite revision checks
# prevent stale reuse at every corpus size.
_CACHEABLE_CORPUS = _TTL_MIN_CORPUS_DOCUMENTS
_SMALL_CORPUS = 12

_UBIQUITOUS_TOKEN = "alpha"
# Present in exactly half the corpus: the drop threshold is inclusive, so this
# token must be retained while a rarer one must not.
_BOUNDARY_TOKEN = "beta"
_RARE_TOKEN = "gamma"

_INSERT_USER_SQL = """
    INSERT INTO users (id, created_at, updated_at)
    VALUES (?, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')
"""
_INSERT_LIFECYCLE_SQL = """
    INSERT INTO user_lifecycles (
        user_id, lifecycle_epoch, lifecycle_cleanup_key, created_at, updated_at
    ) VALUES (?, ?, ?, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')
"""
_INSERT_MEMORY_SQL = """
    INSERT INTO memory_objects (
        id, user_id, canonical_text, object_type, scope, source_kind,
        confidence, created_at, updated_at
    ) VALUES (?, ?, ?, 'evidence', 'user', 'extracted', 0.9,
              '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')
"""


class _CorpusScanCounter:
    """Counts executions of the corpus-scan statement on a live connection."""

    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._original = connection.execute
        self.scans = 0
        connection.execute = self._execute  # type: ignore[method-assign]

    async def _execute(
        self,
        sql: str,
        parameters: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> aiosqlite.Cursor:
        if sql == _CORPUS_SCAN_SQL:
            self.scans += 1
        if parameters is None:
            return await self._original(sql, *args, **kwargs)
        return await self._original(sql, parameters, *args, **kwargs)


def _document_text(index: int, *, marker: str) -> str:
    """One document whose tokens straddle the drop threshold deliberately."""
    tokens = [_UBIQUITOUS_TOKEN, marker, f"distinct{index}"]
    if index % 2:
        tokens.append(_BOUNDARY_TOKEN)
    elif index % 4 == 0:
        tokens.append(_RARE_TOKEN)
    return " ".join(tokens)


async def _build_corpus(
    *,
    user_ids: tuple[str, ...] = ("usr_1",),
    documents_per_user: int = _CACHEABLE_CORPUS,
    markers: dict[str, str] | None = None,
) -> tuple[aiosqlite.Connection, FrozenClock]:
    """Build a corpus by bulk insert.

    These tests assert on scan counts, and the scan runs before any FTS query,
    so the rows only have to exist and belong to the right user.
    """
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 3, 30, 20, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(
        connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock
    )
    await connection.executemany(
        _INSERT_USER_SQL, [(user_id,) for user_id in user_ids]
    )
    # Production creates the canonical user and its immutable lifecycle in the
    # same operation. Keep the fixture faithful so revision triggers always
    # have their lifecycle-scoped authority row.
    await connection.executemany(
        _INSERT_LIFECYCLE_SQL,
        [
            (user_id, f"ule_{user_id}", f"ulk_{user_id}")
            for user_id in user_ids
        ],
    )
    await connection.executemany(
        _INSERT_MEMORY_SQL,
        [
            (
                f"mem_{user_id}_{index}",
                user_id,
                _document_text(index, marker=(markers or {}).get(user_id, "shared")),
            )
            for user_id in user_ids
            for index in range(documents_per_user)
        ],
    )
    await connection.commit()
    return connection, clock


def _plan(sub_query: str) -> RetrievalPlan:
    return RetrievalPlan(
        original_query=sub_query,
        assistant_mode_id="coding_debug",
        workspace_id=None,
        conversation_id="cnv_1",
        fts_queries=build_retrieval_fts_queries(sub_query),
        sub_query_plans=[
            PlannedSubQuery(
                text=sub_query,
                fts_queries=build_retrieval_fts_queries(sub_query),
            )
        ],
        query_type="default",
        scope_filter=[MemoryScope.GLOBAL_USER],
        status_filter=[MemoryStatus.ACTIVE],
        max_candidates=10,
        max_context_items=8,
        privacy_ceiling=1,
        retrieval_levels=[0],
        require_evidence_regrounding=False,
        skip_retrieval=False,
    )


def _pipeline_settings() -> Settings:
    return Settings(
        sqlite_path=":memory:",
        migrations_path=str(MIGRATIONS_DIR),
        manifests_path=str(MANIFESTS_DIR),
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        openai_api_key="test-openai-key",
        openrouter_api_key=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
        llm_chat_model="openai/test-model",
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
        allow_insecure_http=True,
    )


async def _run_turn(
    connection: aiosqlite.Connection,
    clock: FrozenClock,
    cache: TokenDocumentFrequencyCache,
    user_id: str = "usr_1",
) -> None:
    """One turn, built exactly the way production builds one."""
    search = CandidateSearch(
        connection,
        clock,
        token_document_frequency_cache=cache,
    )
    await search.search(_plan(f"{_UBIQUITOUS_TOKEN} distinct3"), user_id)


@pytest.mark.asyncio
async def test_consecutive_turns_scan_the_corpus_once() -> None:
    """The invariant: a second turn for the same user reuses the statistics.

    Each turn builds its own ``CandidateSearch`` exactly as production does, so
    this fails the moment the cache's lifetime is reduced to one request again.
    """
    connection, clock = await _build_corpus()
    counter = _CorpusScanCounter(connection)
    cache = TokenDocumentFrequencyCache()
    try:
        await _run_turn(connection, clock, cache)
        await _run_turn(connection, clock, cache)
        assert counter.scans == 1
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_corpus_revision_invalidates_every_boundary_crossing_mutation() -> None:
    """A single insert, update, or delete can cross the 0.5 boundary.

    ``beta`` starts in exactly 250/500 documents and is therefore filtered.
    A document without it moves the ratio to 250/501, updating that document
    moves it to 251/501, and deleting it restores 250/500. A TTL-only cache
    returns the first decision throughout; the SQLite revision must force a
    fresh scan after all three mutation shapes.
    """
    connection, clock = await _build_corpus()
    counter = _CorpusScanCounter(connection)
    cache = TokenDocumentFrequencyCache()
    try:
        before = await cache.ubiquitous_token_ratios(connection, clock, "usr_1")
        assert before[_BOUNDARY_TOKEN] == pytest.approx(TOKEN_DOC_RATIO_MAX_RATIO)

        await connection.execute(
            _INSERT_MEMORY_SQL,
            (
                "mem_usr_1_boundary_shift",
                "usr_1",
                "alpha shared boundaryshift",
            ),
        )
        await connection.commit()

        after = await cache.ubiquitous_token_ratios(connection, clock, "usr_1")
        assert _BOUNDARY_TOKEN not in after

        await connection.execute(
            """
            UPDATE memory_objects
            SET canonical_text = ?
            WHERE user_id = ? AND id = ?
            """,
            (
                "alpha shared beta boundaryshift",
                "usr_1",
                "mem_usr_1_boundary_shift",
            ),
        )
        await connection.commit()
        after_update = await cache.ubiquitous_token_ratios(
            connection,
            clock,
            "usr_1",
        )
        assert after_update[_BOUNDARY_TOKEN] > TOKEN_DOC_RATIO_MAX_RATIO

        await connection.execute(
            """
            DELETE FROM memory_objects
            WHERE user_id = ? AND id = ?
            """,
            ("usr_1", "mem_usr_1_boundary_shift"),
        )
        await connection.commit()
        after_delete = await cache.ubiquitous_token_ratios(
            connection,
            clock,
            "usr_1",
        )
        assert after_delete[_BOUNDARY_TOKEN] == pytest.approx(
            TOKEN_DOC_RATIO_MAX_RATIO
        )
        assert counter.scans == 4
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_lifecycle_epoch_prevents_revision_aba_reuse() -> None:
    """The same revision number in a new lifecycle is not the same corpus."""
    connection, clock = await _build_corpus()
    counter = _CorpusScanCounter(connection)
    cache = TokenDocumentFrequencyCache()
    try:
        before = await cache.ubiquitous_token_ratios(connection, clock, "usr_1")
        assert "shared" in before

        # Retire the lifecycle, then rebuild a same-sized corpus. Both lifecycle
        # counters end at the same numeric value, so revision alone would reuse
        # the old entry until TTL expiry.
        await connection.execute(
            "DELETE FROM memory_objects WHERE user_id = ?",
            ("usr_1",),
        )
        await connection.execute(
            "DELETE FROM user_lifecycles WHERE user_id = ?",
            ("usr_1",),
        )
        await connection.execute(
            _INSERT_LIFECYCLE_SQL,
            ("usr_1", "ule_usr_1_replacement", "ulk_usr_1_replacement"),
        )
        await connection.executemany(
            _INSERT_MEMORY_SQL,
            [
                (
                    f"mem_usr_1_replacement_{index}",
                    "usr_1",
                    _document_text(index, marker="replacement"),
                )
                for index in range(_CACHEABLE_CORPUS)
            ],
        )
        await connection.commit()

        after = await cache.ubiquitous_token_ratios(connection, clock, "usr_1")
        assert "replacement" in after
        assert "shared" not in after
        assert counter.scans == 2
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_scan_is_not_published_when_erasure_changes_revision_mid_scan() -> None:
    """A cross-process-style erasure between scan and publish forces a retry."""
    connection, clock = await _build_corpus(documents_per_user=_SMALL_CORPUS)
    original_execute = connection.execute
    scan_count = 0
    erase_after_first_scan = True

    async def intercept_execute(
        sql: str,
        parameters: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        nonlocal scan_count
        if parameters is None:
            cursor = await original_execute(sql, *args, **kwargs)
        else:
            cursor = await original_execute(sql, parameters, *args, **kwargs)
        if sql != _CORPUS_SCAN_SQL:
            return cursor
        scan_count += 1

        class ErasingCursor:
            async def fetchall(self) -> Any:
                nonlocal erase_after_first_scan
                rows = await cursor.fetchall()
                if erase_after_first_scan:
                    erase_after_first_scan = False
                    await original_execute(
                        "DELETE FROM memory_objects WHERE user_id = ?",
                        ("usr_1",),
                    )
                    await connection.commit()
                return rows

        return ErasingCursor()

    connection.execute = intercept_execute  # type: ignore[method-assign]
    cache = TokenDocumentFrequencyCache()
    try:
        ratios = await cache.ubiquitous_token_ratios(connection, clock, "usr_1")

        assert ratios == {}
        assert scan_count == 2
        assert cache._entries["usr_1"].total_documents == 0  # noqa: SLF001
    finally:
        connection.execute = original_execute  # type: ignore[method-assign]
        await connection.close()


@pytest.mark.asyncio
async def test_corpus_without_lifecycle_revision_authority_fails_fast() -> None:
    """Orphaned rows must not degrade into an unversioned reusable cache."""
    connection, clock = await _build_corpus()
    try:
        await connection.execute(
            "DELETE FROM user_lifecycles WHERE user_id = ?",
            ("usr_1",),
        )
        await connection.commit()

        with pytest.raises(RuntimeError, match="no lifecycle-scoped revision"):
            await TokenDocumentFrequencyCache().ubiquitous_token_ratios(
                connection,
                clock,
                "usr_1",
            )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_per_request_cache_rescans_every_turn() -> None:
    """Control for the counter: an unshared cache pays the scan on every turn.

    This is the behavior the previous implementation had, and it proves the
    assertion above measures scans rather than an always-true condition.
    """
    connection, clock = await _build_corpus()
    counter = _CorpusScanCounter(connection)
    try:
        await _run_turn(connection, clock, TokenDocumentFrequencyCache())
        await _run_turn(connection, clock, TokenDocumentFrequencyCache())
        assert counter.scans == 2
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_small_corpus_is_rescanned_when_caching_has_no_benefit() -> None:
    """The TTL is not applied where repeating the scan is already cheap."""
    connection, clock = await _build_corpus(documents_per_user=_SMALL_CORPUS)
    counter = _CorpusScanCounter(connection)
    cache = TokenDocumentFrequencyCache()
    try:
        await _run_turn(connection, clock, cache)
        await _run_turn(connection, clock, cache)
        assert counter.scans == 2
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_statistics_are_recomputed_once_the_entry_expires() -> None:
    connection, clock = await _build_corpus()
    counter = _CorpusScanCounter(connection)
    cache = TokenDocumentFrequencyCache()
    try:
        await _run_turn(connection, clock, cache)
        clock.advance(seconds=_CACHE_TTL_SECONDS - 1.0)
        await _run_turn(connection, clock, cache)
        assert counter.scans == 1
        clock.advance(seconds=2.0)
        await _run_turn(connection, clock, cache)
        assert counter.scans == 2
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_statistics_are_partitioned_per_user() -> None:
    """One user's corpus never supplies another user's statistics."""
    connection, clock = await _build_corpus(
        user_ids=("usr_1", "usr_2"),
        markers={"usr_1": "onlyfirst", "usr_2": "onlysecond"},
    )
    counter = _CorpusScanCounter(connection)
    cache = TokenDocumentFrequencyCache()
    try:
        first = await cache.ubiquitous_token_ratios(connection, clock, "usr_1")
        second = await cache.ubiquitous_token_ratios(connection, clock, "usr_2")
        assert counter.scans == 2
        # Each marker is ubiquitous in its owner's corpus and absent from the
        # other's, so a leak in either direction changes these mappings.
        assert "onlyfirst" in first
        assert "onlyfirst" not in second
        assert "onlysecond" in second
        assert "onlysecond" not in first
        empty = await cache.ubiquitous_token_ratios(connection, clock, "usr_absent")
        assert empty == {}
        assert counter.scans == 3
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_only_corpus_ubiquitous_tokens_are_retained() -> None:
    """Entries hold the tokens a query is filtered on, not the vocabulary."""
    connection, clock = await _build_corpus()
    cache = TokenDocumentFrequencyCache()
    try:
        ratios = await cache.ubiquitous_token_ratios(connection, clock, "usr_1")
        assert ratios[_UBIQUITOUS_TOKEN] == pytest.approx(1.0)
        assert ratios[_BOUNDARY_TOKEN] == pytest.approx(TOKEN_DOC_RATIO_MAX_RATIO)
        assert "distinct3" not in ratios
        assert _RARE_TOKEN not in ratios
        assert all(ratio >= TOKEN_DOC_RATIO_MAX_RATIO for ratio in ratios.values())
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_returned_statistics_cannot_be_mutated_by_a_consumer() -> None:
    """Entries are handed out by reference, so they must be read-only."""
    connection, clock = await _build_corpus()
    cache = TokenDocumentFrequencyCache()
    try:
        ratios = await cache.ubiquitous_token_ratios(connection, clock, "usr_1")
        with pytest.raises(TypeError):
            cast(Any, ratios)[_UBIQUITOUS_TOKEN] = 0.0
        with pytest.raises(TypeError):
            del cast(Any, ratios)[_UBIQUITOUS_TOKEN]
        # The in-place mutators are not merely guarded, they are absent.
        assert not hasattr(ratios, "clear")
        assert not hasattr(ratios, "update")
        empty = await cache.ubiquitous_token_ratios(connection, clock, "usr_absent")
        with pytest.raises(TypeError):
            cast(Any, empty)["injected"] = 1.0
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_forget_drops_statistics_derived_from_erased_content() -> None:
    """Erasure removes derived statistics from process memory immediately."""
    connection, clock = await _build_corpus()
    counter = _CorpusScanCounter(connection)
    cache = TokenDocumentFrequencyCache()
    try:
        await cache.ubiquitous_token_ratios(connection, clock, "usr_1")
        assert counter.scans == 1
        cache.forget("usr_1")
        await cache.ubiquitous_token_ratios(connection, clock, "usr_1")
        assert counter.scans == 2
        # Forgetting an absent user is not an error.
        cache.forget("usr_never_seen")
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_corpus_scan_failure_propagates_instead_of_degrading_filtering() -> None:
    """A broken scan must raise, not silently disable token filtering.

    The scan used to swallow ``OperationalError`` and return "no statistics",
    which turned a hard database failure into quiet recall degradation.
    """
    connection, clock = await _build_corpus(documents_per_user=_SMALL_CORPUS)
    original = connection.execute

    async def failing_execute(
        sql: str,
        parameters: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> aiosqlite.Cursor:
        if sql == _CORPUS_SCAN_SQL:
            raise aiosqlite.OperationalError("no such table: memory_objects")
        if parameters is None:
            return await original(sql, *args, **kwargs)
        return await original(sql, parameters, *args, **kwargs)

    connection.execute = failing_execute  # type: ignore[method-assign]
    try:
        with pytest.raises(aiosqlite.OperationalError):
            await _run_turn(connection, clock, TokenDocumentFrequencyCache())
    finally:
        connection.execute = original  # type: ignore[method-assign]
        await connection.close()


@pytest.mark.asyncio
async def test_cache_evicts_the_least_recently_used_user() -> None:
    """The map is bounded: a multi-tenant process cannot grow it forever."""
    bulk_users = tuple(f"usr_bulk_{index}" for index in range(_MAX_CACHED_USERS + 1))
    connection, clock = await _build_corpus(user_ids=bulk_users)
    counter = _CorpusScanCounter(connection)
    cache = TokenDocumentFrequencyCache()
    try:
        for user_id in bulk_users:
            await cache.ubiquitous_token_ratios(connection, clock, user_id)
        assert counter.scans == len(bulk_users)
        # The first user seen was evicted when the last one was admitted.
        await cache.ubiquitous_token_ratios(connection, clock, bulk_users[0])
        assert counter.scans == len(bulk_users) + 1
        # Re-admitting it evicted the next-oldest in turn, but not the one after.
        await cache.ubiquitous_token_ratios(connection, clock, bulk_users[2])
        assert counter.scans == len(bulk_users) + 1
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_pipeline_hands_its_cache_to_candidate_search() -> None:
    """The pipeline must forward the runtime cache, never allocate its own.

    ``RetrievalPipeline`` is built once per request, so a cache created inside
    it would be exactly the defect this module exists to prevent.
    """
    connection, clock = await _build_corpus(documents_per_user=_SMALL_CORPUS)
    cache = TokenDocumentFrequencyCache()
    try:
        pipelines = [
            RetrievalPipeline(
                connection=connection,
                llm_client=cast(Any, object()),
                embedding_index=NoneBackend(),
                clock=clock,
                token_document_frequency_cache=cache,
                settings=_pipeline_settings(),
            )
            for _turn in range(2)
        ]
        for pipeline in pipelines:
            search = pipeline._candidate_search  # noqa: SLF001
            assert search._token_document_frequency_cache is cache  # noqa: SLF001
    finally:
        await connection.close()


def test_runtime_owns_one_cache_for_the_process() -> None:
    """``AppRuntime`` is the long-lived, one-per-database owner."""
    assert "token_document_frequency_cache" in {
        field.name for field in fields(AppRuntime)
    }
    declared = AppRuntime.__dataclass_fields__["token_document_frequency_cache"]
    assert declared.default_factory is TokenDocumentFrequencyCache
