"""``memory_objects.queryable_at``: when a memory first became retrievable.

Migration 0069 measured ingest latency as ``created_at -
source_message_created_at`` and admitted in its own comment that the left-hand
side was only right for rows born ``active``: retrieval filters candidates on
``status = 'active'``, and extraction routinely mints ``review_required`` and
``pending_user_confirmation`` rows that are not retrievable then, and possibly
never will be. 0071 replaces the guess with a real stamp, and these tests pin the
claims that make it worth trusting: it is written when a memory is born
queryable, written on the transition INTO queryable, never written for a memory
that never got there, never rewritten once set, and never invented for a row
that reached ``active`` before the column existed. The last test scans the
package so a new status write cannot quietly skip all of that.
"""

from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path
import re
from shutil import copy2
from typing import Any

import aiosqlite
import pytest

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.consent_repository import PendingMemoryConfirmationRepository
from atagia.core.db_sqlite import MigrationManager, initialize_database
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    UserRepository,
)
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.models.schemas_memory import (
    IntimacyBoundary,
    MemoryCategory,
    MemoryObjectType,
    MemoryScope,
    MemorySensitivity,
    MemorySourceKind,
    MemoryStatus,
    SummaryViewKind,
)
from atagia.services.confirmation_service import PendingConfirmationService
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMProvider,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS_DIR = REPO_ROOT / "src" / "atagia" / "resources" / "migrations"
MANIFESTS_DIR = REPO_ROOT / "src" / "atagia" / "resources" / "manifests"
PACKAGE_DIR = REPO_ROOT / "src" / "atagia"

QUERYABLE_AT_MIGRATION_VERSION = 71
USER_ID = "usr_queryable_at"
CONVERSATION_ID = "cnv_queryable_at"
START = datetime(2026, 7, 25, 9, 0, tzinfo=timezone.utc)


class _ConsentIntentProvider(LLMProvider):
    """Answers the consent-intent classification with a fixed verdict."""

    name = "queryable-at-tests"

    def __init__(self, intent: str) -> None:
        self.intent = intent

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=f'{{"intent":"{self.intent}"}}',
        )


def _settings() -> Settings:
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
        llm_chat_model="reply-test-model",
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
        allow_insecure_http=True,
    )


async def _seed_namespace(
    connection: aiosqlite.Connection,
    clock: FrozenClock,
) -> None:
    await sync_assistant_modes(
        connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock
    )
    await UserRepository(connection, clock).create_user(USER_ID)
    await ConversationRepository(connection, clock).create_conversation(
        CONVERSATION_ID,
        USER_ID,
        None,
        "personal_assistant",
        "Chat",
    )


async def _create_gated_memory(
    memories: MemoryObjectRepository,
    confirmations: PendingMemoryConfirmationRepository,
    clock: FrozenClock,
    *,
    memory_id: str,
) -> dict[str, Any]:
    """Create a consent-gated memory and the marker the user will answer."""
    pending = await memories.create_memory_object(
        memory_id=memory_id,
        user_id=USER_ID,
        conversation_id=CONVERSATION_ID,
        assistant_mode_id="personal_assistant",
        object_type=MemoryObjectType.EVIDENCE,
        scope=MemoryScope.GLOBAL_USER,
        canonical_text="Banking card PIN: 4512",
        index_text="bank card PIN",
        source_kind=MemorySourceKind.EXTRACTED,
        confidence=0.97,
        privacy_level=3,
        memory_category=MemoryCategory.PIN_OR_PASSWORD,
        preserve_verbatim=True,
        status=MemoryStatus.PENDING_USER_CONFIRMATION,
        commit=False,
    )
    await confirmations.create_marker(
        user_id=USER_ID,
        conversation_id=CONVERSATION_ID,
        memory_id=memory_id,
        category=MemoryCategory.PIN_OR_PASSWORD,
        created_at=str(pending["created_at"]),
        intended_scope=MemoryScope.USER,
        intended_sensitivity=MemorySensitivity.SECRET,
        policy_snapshot={"source": "test"},
        policy_proven=True,
        commit=False,
    )
    await confirmations.mark_markers_asked(
        USER_ID,
        [memory_id],
        asked_at=clock.now().isoformat(),
        commit=False,
    )
    return pending


async def _answer_confirmation(
    connection: aiosqlite.Connection,
    clock: FrozenClock,
    *,
    intent: str,
    message_text: str,
) -> None:
    provider = _ConsentIntentProvider(intent)
    service = PendingConfirmationService(
        connection,
        clock,
        llm_client=LLMClient(provider_name=provider.name, providers=[provider]),
        settings=_settings(),
    )
    plan = await service.plan_turn(
        user_id=USER_ID,
        conversation_id=CONVERSATION_ID,
        message_text=message_text,
    )
    await service.apply_turn_plan(user_id=USER_ID, plan=plan, commit=True)


# ---------------------------------------------------------------------------
# Stamping behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_born_active_is_stamped_at_creation() -> None:
    """Born active means queryable at birth, and the two stamps must agree.

    The FTS trigger fires in the same statement that inserts the row and the row
    is already retrieval-eligible, so ``created_at`` genuinely is the moment it
    became queryable. This is the one case 0069's original reading got right, and
    the equality here is what keeps it right instead of merely close.
    """
    clock = FrozenClock(START)
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    try:
        await _seed_namespace(connection, clock)
        memories = MemoryObjectRepository(connection, clock)
        created = await memories.create_memory_object(
            user_id=USER_ID,
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="The user cycles to work on Tuesdays.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.9,
            privacy_level=0,
        )
        assert created["status"] == MemoryStatus.ACTIVE.value
        assert created["queryable_at"] == created["created_at"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_memory_born_gated_is_not_stamped_at_creation() -> None:
    """A pending row is in the FTS index but not retrievable, so no stamp."""
    clock = FrozenClock(START)
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    try:
        await _seed_namespace(connection, clock)
        memories = MemoryObjectRepository(connection, clock)
        for status in (
            MemoryStatus.PENDING_USER_CONFIRMATION,
            MemoryStatus.REVIEW_REQUIRED,
        ):
            created = await memories.create_memory_object(
                user_id=USER_ID,
                object_type=MemoryObjectType.EVIDENCE,
                scope=MemoryScope.GLOBAL_USER,
                canonical_text=f"Gated memory ({status.value}).",
                source_kind=MemorySourceKind.EXTRACTED,
                confidence=0.9,
                privacy_level=0,
                status=status,
            )
            assert created["queryable_at"] is None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_confirmed_memory_is_stamped_when_it_becomes_retrievable() -> None:
    """The stamp follows the transition, not the row's birth.

    This runs the real consent flow: a gated memory sits unretrievable for an
    hour, the user says yes, and ``PendingConfirmationService`` activates it. The
    stamp has to be the moment of activation, and the gap it opens against
    ``created_at`` is precisely the dishonesty 0071 exists to remove.
    """
    clock = FrozenClock(START)
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    try:
        await _seed_namespace(connection, clock)
        memories = MemoryObjectRepository(connection, clock)
        confirmations = PendingMemoryConfirmationRepository(connection, clock)
        pending = await _create_gated_memory(
            memories, confirmations, clock, memory_id="mem_confirmed"
        )
        await connection.commit()
        assert pending["queryable_at"] is None

        clock.advance(seconds=3600)
        await _answer_confirmation(
            connection, clock, intent="confirm", message_text="yes"
        )

        updated = await memories.get_memory_object("mem_confirmed", USER_ID)
        assert updated is not None
        assert updated["status"] == MemoryStatus.ACTIVE.value
        assert updated["queryable_at"] == clock.now().isoformat()
        assert updated["queryable_at"] > updated["created_at"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_declined_memory_never_gets_a_stamp() -> None:
    """NULL forever is a true statement: this memory never became retrievable."""
    clock = FrozenClock(START)
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    try:
        await _seed_namespace(connection, clock)
        memories = MemoryObjectRepository(connection, clock)
        confirmations = PendingMemoryConfirmationRepository(connection, clock)
        await _create_gated_memory(
            memories, confirmations, clock, memory_id="mem_declined"
        )
        await connection.commit()

        clock.advance(seconds=3600)
        await _answer_confirmation(
            connection, clock, intent="deny", message_text="no, forget it"
        )

        updated = await memories.get_memory_object("mem_declined", USER_ID)
        assert updated is not None
        assert updated["status"] == MemoryStatus.DECLINED.value
        assert updated["queryable_at"] is None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_reactivation_keeps_the_first_stamp() -> None:
    """``queryable_at`` is the FIRST time the memory was queryable, not the last.

    A row that goes active -> archived -> active WAS retrievable at the first
    activation, so overwriting the stamp would restate an old ingest as a fresh
    one and make ingest-latency averages drift with archival churn.
    """
    clock = FrozenClock(START)
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    try:
        await _seed_namespace(connection, clock)
        memories = MemoryObjectRepository(connection, clock)
        created = await memories.create_memory_object(
            memory_id="mem_recycled",
            user_id=USER_ID,
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="The user prefers window seats.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.9,
            privacy_level=0,
        )
        first_stamp = created["queryable_at"]
        assert first_stamp is not None

        clock.advance(seconds=3600)
        assert await memories.archive_memory_object("mem_recycled", USER_ID) is True
        archived = await memories.get_memory_object("mem_recycled", USER_ID)
        assert archived is not None
        assert archived["status"] == MemoryStatus.ARCHIVED.value
        # Leaving retrievability does not erase the fact that it was reached.
        assert archived["queryable_at"] == first_stamp

        clock.advance(seconds=3600)
        reactivated = await memories.update_memory_object_status(
            memory_id="mem_recycled",
            user_id=USER_ID,
            status=MemoryStatus.ACTIVE,
            expected_current_status=MemoryStatus.ARCHIVED,
        )
        assert reactivated is not None
        assert reactivated["status"] == MemoryStatus.ACTIVE.value
        assert reactivated["queryable_at"] == first_stamp
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_status_rewrite_never_stamps_a_row_that_predates_the_column() -> None:
    """A pre-0071 active row keeps NULL; "now" would be a fabricated measurement.

    The row below is what an upgraded database is full of: active, retrievable,
    and carrying no record of when that started. Every statement that rewrites
    its status has to leave the column alone, because the only value it could
    write is a lie about when the memory became queryable.
    """
    clock = FrozenClock(START)
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    try:
        await _seed_namespace(connection, clock)
        memories = MemoryObjectRepository(connection, clock)
        await memories.create_memory_object(
            memory_id="mem_legacy",
            user_id=USER_ID,
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="Written before the column existed.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.9,
            privacy_level=0,
        )
        # Reproduce the upgraded-database state the migration deliberately leaves.
        await connection.execute(
            "UPDATE memory_objects SET queryable_at = NULL WHERE id = ? AND user_id = ?",
            ("mem_legacy", USER_ID),
        )
        await connection.commit()

        clock.advance(seconds=3600)
        rewritten = await memories.update_memory_object_status(
            memory_id="mem_legacy",
            user_id=USER_ID,
            status=MemoryStatus.ACTIVE,
            expected_current_status=MemoryStatus.ACTIVE,
        )
        assert rewritten is not None
        assert rewritten["status"] == MemoryStatus.ACTIVE.value
        assert rewritten["queryable_at"] is None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_summary_mirror_keeps_its_stamp_across_recompaction() -> None:
    """The mirror's status is rewritten on every compaction, its stamp is not.

    ``upsert_summary_mirror`` restates ``status = 'active'`` each time the
    compactor refreshes a view. Without the pre-update read on the right-hand
    side, every recompaction would restamp a mirror that has been retrievable
    since it was first written.
    """
    clock = FrozenClock(START)
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    try:
        await _seed_namespace(connection, clock)
        memories = MemoryObjectRepository(connection, clock)
        created = await memories.upsert_summary_mirror(
            user_id=USER_ID,
            summary_view_id="sum_queryable_at",
            summary_kind=SummaryViewKind.EPISODE,
            hierarchy_level=0,
            summary_text="The user spent the morning on retry handling.",
            source_object_ids=[],
            created_at=clock.now().isoformat(),
        )
        first_stamp = created["queryable_at"]
        assert first_stamp == created["created_at"]

        clock.advance(seconds=1800)
        recompacted = await memories.upsert_summary_mirror(
            user_id=USER_ID,
            summary_view_id="sum_queryable_at",
            summary_kind=SummaryViewKind.EPISODE,
            hierarchy_level=0,
            summary_text="The user spent the morning and afternoon on retry handling.",
            source_object_ids=[],
            created_at=clock.now().isoformat(),
        )
        assert recompacted["updated_at"] == clock.now().isoformat()
        assert recompacted["queryable_at"] == first_stamp
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_tightening_write_restrictions_does_not_touch_the_stamp() -> None:
    """A dedupe hit rewrites the status it read; the stamp must survive intact.

    ``merge_memory_object_write_restrictions`` carries a status derived from the
    existing row, so it is one of the statements whose target is chosen at
    runtime. Tightening an active row to ``review_required`` takes it out of
    retrievability, which is not a reason to forget that it was retrievable.
    """
    clock = FrozenClock(START)
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    try:
        await _seed_namespace(connection, clock)
        memories = MemoryObjectRepository(connection, clock)
        created = await memories.create_memory_object(
            memory_id="mem_tightened",
            user_id=USER_ID,
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="The user's building code is on the fridge.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.9,
            privacy_level=0,
        )
        first_stamp = created["queryable_at"]
        assert first_stamp is not None

        clock.advance(seconds=1800)
        tightened = await memories.merge_memory_object_write_restrictions(
            user_id=USER_ID,
            memory_id="mem_tightened",
            privacy_level=3,
            intimacy_boundary=IntimacyBoundary.ORDINARY,
            intimacy_boundary_confidence=0.0,
            sensitivity=MemorySensitivity.SECRET,
            themes=[],
            auto_expires=False,
            platform_locked=False,
            platform_id_lock=None,
            review_required=True,
        )
        assert tightened["status"] == MemoryStatus.REVIEW_REQUIRED.value
        assert tightened["queryable_at"] == first_stamp
    finally:
        await connection.close()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_queryable_at_migration_applies_on_a_fresh_database() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    try:
        cursor = await connection.execute("PRAGMA table_info(memory_objects)")
        columns = {row["name"] for row in await cursor.fetchall()}
        assert "queryable_at" in columns
        # Both ends of the ingest interval, or the latency is unmeasurable.
        assert "source_message_created_at" in columns
        cursor = await connection.execute(
            "SELECT applied_at FROM schema_migrations WHERE version = ?",
            (QUERYABLE_AT_MIGRATION_VERSION,),
        )
        # The NULL disambiguation in 0071's comment leans on this row existing:
        # inside the window a NULL means "never became queryable", outside it
        # means "predates the column".
        assert await cursor.fetchone() is not None
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_queryable_at_migration_upgrades_a_database_at_the_previous_head(
    tmp_path: Path,
) -> None:
    """0071 lands on a database already carrying 0069/0070's telemetry columns.

    The pre-existing rows are the point: an active one and a gated one, neither
    of which can be honestly backfilled, both of which must come out NULL rather
    than inherit ``created_at``.
    """
    bootstrap = tmp_path / "migrations-through-0070"
    bootstrap.mkdir()
    manager = MigrationManager(MIGRATIONS_DIR)
    for migration in manager.discover():
        if migration.version < QUERYABLE_AT_MIGRATION_VERSION:
            copy2(migration.path, bootstrap / migration.path.name)
    database_path = tmp_path / "queryable-at-upgrade.db"

    clock = FrozenClock(START)
    connection = await initialize_database(str(database_path), bootstrap)
    try:
        cursor = await connection.execute("PRAGMA table_info(memory_objects)")
        columns = {row["name"] for row in await cursor.fetchall()}
        assert "queryable_at" not in columns
        await _seed_namespace(connection, clock)
        # Written with raw SQL on purpose: the repository writes the column that
        # does not exist yet, which is exactly the state an upgraded database
        # arrives in.
        timestamp = clock.now().isoformat()
        for memory_id, status in (
            ("mem_legacy_active", MemoryStatus.ACTIVE),
            ("mem_legacy_gated", MemoryStatus.REVIEW_REQUIRED),
        ):
            await connection.execute(
                """
                INSERT INTO memory_objects(
                    id, user_id, object_type, scope, scope_canonical,
                    canonical_text, source_kind, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    USER_ID,
                    MemoryObjectType.EVIDENCE.value,
                    MemoryScope.USER.value,
                    MemoryScope.USER.value,
                    f"Legacy row ({status.value}).",
                    MemorySourceKind.EXTRACTED.value,
                    status.value,
                    timestamp,
                    timestamp,
                ),
            )
        await connection.commit()
    finally:
        await connection.close()

    connection = await initialize_database(str(database_path), MIGRATIONS_DIR)
    try:
        cursor = await connection.execute(
            "SELECT id, status, queryable_at FROM memory_objects WHERE user_id = ? ORDER BY id",
            (USER_ID,),
        )
        rows = {row["id"]: row for row in await cursor.fetchall()}
        # No backfill, not even for the active row: it genuinely was queryable,
        # but nothing recorded when, and created_at is exactly the guess 0069
        # refused to trust.
        assert rows["mem_legacy_active"]["status"] == MemoryStatus.ACTIVE.value
        assert rows["mem_legacy_active"]["queryable_at"] is None
        assert rows["mem_legacy_gated"]["queryable_at"] is None

        # The column is live from here on: a memory written after the upgrade is
        # stamped like any other.
        clock.advance(seconds=3600)
        fresh = await MemoryObjectRepository(connection, clock).create_memory_object(
            user_id=USER_ID,
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="Written after the upgrade.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.9,
            privacy_level=0,
        )
        assert fresh["queryable_at"] == fresh["created_at"]
    finally:
        await connection.close()


# ---------------------------------------------------------------------------
# Write-site inventory
# ---------------------------------------------------------------------------


# Every statement that writes memory_objects.status but NOT queryable_at, with
# the reason it needs no stamp: each writes a hardcoded status that is never
# 'active', so it can only move a row out of retrievability. A statement whose
# target status is chosen at runtime does not belong here -- it belongs in the
# covered set, because a missed transition silently produces a NULL that reads
# as "never became queryable", which is the exact dishonesty 0071 removes.
_STATUS_WRITES_WITHOUT_A_STAMP: dict[tuple[str, str], str] = {
    (
        "core/repositories.py",
        "MemoryObjectRepository.archive_memory_object",
    ): "active -> archived",
    (
        "memory/belief_reviser.py",
        "BeliefReviser._apply_supersede",
    ): "-> superseded",
    (
        "memory/belief_reviser.py",
        "BeliefReviser._archive_belief",
    ): "-> archived",
    (
        "memory/lifecycle.py",
        "MemoryLifecycleManager._archive_low_value_memories",
    ): "active -> archived",
    (
        "memory/lifecycle.py",
        "MemoryLifecycleManager._archive_expired_state_snapshots",
    ): "active -> archived",
    (
        "memory/lifecycle.py",
        "MemoryLifecycleManager._decline_expired_pending_confirmations",
    ): "pending_user_confirmation -> declined on TTL expiry",
    (
        "services/lifecycle_service.py",
        "ConversationLifecycleService._archive_conversation_guarded",
    ): "-> archived",
    (
        "services/lifecycle_service.py",
        "ConversationLifecycleService._archive_memory",
    ): "-> archived",
    (
        "services/lifecycle_service.py",
        "ConversationLifecycleService._tombstone_memory_rows",
    ): "-> deleted",
    (
        "services/sidecar_service.py",
        "SidecarService._mark_broad_conversation_rows_review_only",
    ): "-> review_required",
}


def _sql_literals(node: ast.AST) -> list[str]:
    """Every string literal in `node`, with f-string fragments joined."""
    nested = {
        id(part)
        for child in ast.walk(node)
        if isinstance(child, ast.JoinedStr)
        for part in child.values
    }
    literals: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.JoinedStr):
            literals.append(
                "".join(
                    part.value
                    for part in child.values
                    if isinstance(part, ast.Constant) and isinstance(part.value, str)
                )
            )
        elif (
            isinstance(child, ast.Constant)
            and isinstance(child.value, str)
            and id(child) not in nested
        ):
            literals.append(child.value)
    return literals


def _collect_memory_status_writes(
    node: ast.AST,
    prefix: str,
    module: str,
    covered: set[tuple[str, str]],
    uncovered: set[tuple[str, str]],
) -> None:
    for child in ast.iter_child_nodes(node):
        if not isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        qualname = f"{prefix}.{child.name}" if prefix else child.name
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
            for literal in _sql_literals(child):
                marker = literal.find("UPDATE memory_objects")
                if marker < 0:
                    continue
                statement = literal[marker:]
                if re.search(r"\bstatus\s*=", statement) is None:
                    continue
                site = (module, qualname)
                if "queryable_at" in statement:
                    covered.add(site)
                else:
                    uncovered.add(site)
        _collect_memory_status_writes(child, qualname, module, covered, uncovered)


def test_every_runtime_chosen_status_write_stamps_queryable_at() -> None:
    """A new status write must be classified, not silently left unstamped.

    Nothing in SQLite forces a writer to maintain ``queryable_at``, so this scan
    is what keeps the invariant from decaying: it fails when a status-writing
    statement appears, moves, or is renamed without a decision being recorded
    here.
    """
    covered: set[tuple[str, str]] = set()
    uncovered: set[tuple[str, str]] = set()
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        module = str(path.relative_to(PACKAGE_DIR))
        _collect_memory_status_writes(
            ast.parse(path.read_text(encoding="utf-8")), "", module, covered, uncovered
        )

    assert uncovered == set(_STATUS_WRITES_WITHOUT_A_STAMP)
    # The runtime-chosen ones, spelled out so a deletion is as visible as an
    # addition. update_memory_object_status is the live confirmation path,
    # upsert_summary_mirror rewrites a mirror's status on every compaction, and
    # merge_memory_object_write_restrictions rewrites a status it derived from
    # the row it found.
    assert covered == {
        ("core/repositories.py", "MemoryObjectRepository.update_memory_object_status"),
        ("core/repositories.py", "MemoryObjectRepository.upsert_summary_mirror"),
        (
            "core/repositories.py",
            "MemoryObjectRepository.merge_memory_object_write_restrictions",
        ),
    }


def test_the_only_memory_object_insert_supplies_queryable_at() -> None:
    """Born-active rows are stamped by the INSERT, so it must carry the column."""
    inserts = [
        literal
        for path in sorted(PACKAGE_DIR.rglob("*.py"))
        for literal in _sql_literals(ast.parse(path.read_text(encoding="utf-8")))
        if "INSERT INTO memory_objects(" in literal
    ]
    assert len(inserts) == 1
    assert "queryable_at" in inserts[0]
