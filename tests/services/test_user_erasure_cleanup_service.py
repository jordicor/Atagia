"""Service-boundary tests for durable right-to-erasure cleanup."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import close_connection, initialize_database, open_connection
from atagia.core.repositories import UserRepository
from atagia.core.storage_backend import (
    InProcessBackend,
    LegacyTransientPurgeResult,
    build_recent_window_key,
)
from atagia.core.user_erasure_repository import UserErasureRepository
from atagia.core.user_lifecycle_repository import UserLifecycleRepository
from atagia.services.errors import (
    UserDeletedError,
    UserErasureCleanupPendingError,
)
from atagia.services.lifecycle_service import (
    ERASE_ALL_DATA_CONFIRMATION,
    ConversationLifecycleService,
)
from atagia.services.user_erasure_cleanup_service import (
    UserErasureCleanupService,
    recover_pending_user_erasures,
)

MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "migrations"
)
CLOCK = FrozenClock(datetime(2026, 7, 13, 15, 0, tzinfo=timezone.utc))


class _FlakyLifecycleBackend(InProcessBackend):
    fail_revoke = True

    def __init__(self) -> None:
        super().__init__()
        self.legacy_purge_calls = 0

    async def revoke_lifecycle_and_purge_notifications(
        self,
        lifecycle_cleanup_key: str,
        lifecycle_epoch: str,
        *,
        group_name: str,
    ) -> int:
        if self.fail_revoke:
            raise ConnectionError("injected transient-backend outage")
        return await super().revoke_lifecycle_and_purge_notifications(
            lifecycle_cleanup_key,
            lifecycle_epoch,
            group_name=group_name,
        )

    async def purge_legacy_transient_state(
        self,
        database_path: str,
        user_id: str,
    ) -> LegacyTransientPurgeResult:
        self.legacy_purge_calls += 1
        return await super().purge_legacy_transient_state(database_path, user_id)


class _PausingLifecycleBackend(_FlakyLifecycleBackend):
    def __init__(self) -> None:
        super().__init__()
        self.pause_next_revoke = False
        self.revoke_paused = asyncio.Event()
        self.resume_revoke = asyncio.Event()

    async def revoke_lifecycle_and_purge_notifications(
        self,
        lifecycle_cleanup_key: str,
        lifecycle_epoch: str,
        *,
        group_name: str,
    ) -> int:
        purged = await super().revoke_lifecycle_and_purge_notifications(
            lifecycle_cleanup_key,
            lifecycle_epoch,
            group_name=group_name,
        )
        if self.pause_next_revoke:
            self.pause_next_revoke = False
            self.revoke_paused.set()
            await self.resume_revoke.wait()
        return purged


class _PausingLegacyPurgeBackend(_FlakyLifecycleBackend):
    def __init__(self) -> None:
        super().__init__()
        self.pause_next_purge = False
        self.purge_paused = asyncio.Event()
        self.resume_purge = asyncio.Event()

    async def purge_legacy_transient_state(
        self,
        database_path: str,
        user_id: str,
    ) -> LegacyTransientPurgeResult:
        if self.pause_next_purge:
            self.pause_next_purge = False
            self.purge_paused.set()
            await self.resume_purge.wait()
        return await super().purge_legacy_transient_state(database_path, user_id)


class _MalformedLegacyBackend(InProcessBackend):
    malformed_candidates = 1

    async def purge_legacy_transient_state(
        self,
        database_path: str,
        user_id: str,
    ) -> LegacyTransientPurgeResult:
        del database_path, user_id
        return LegacyTransientPurgeResult(
            malformed_candidates=self.malformed_candidates,
        )


class _NoopEmbeddingIndex:
    async def delete(self, memory_id: str) -> None:
        del memory_id


def _runtime(database_path: str, backend: InProcessBackend) -> SimpleNamespace:
    return SimpleNamespace(
        settings=SimpleNamespace(
            storage_backend="inprocess",
            erasure_purge_streams=True,
        ),
        clock=CLOCK,
        llm_client=None,
        storage_backend=backend,
        database_path=database_path,
        embedding_index=_NoopEmbeddingIndex(),
    )


@pytest.mark.asyncio
async def test_external_failure_is_resumable_and_startup_recovery_finishes_exact_cleanup(
    tmp_path: Path,
) -> None:
    database_path = str(tmp_path / "durable-erasure.db")
    backend = _FlakyLifecycleBackend()
    connection = await initialize_database(database_path, MIGRATIONS_DIR)
    await UserRepository(connection, CLOCK).create_user("usr_cleanup")
    identity = await UserLifecycleRepository(connection, CLOCK).get_active_identity(
        "usr_cleanup"
    )
    assert identity is not None
    nonce = "prepare-before-erasure"
    assert (
        await backend.prepare_lifecycle_mirror(
            identity.lifecycle_cleanup_key,
            identity.lifecycle_epoch,
            nonce,
        )
        == f"preparing:{identity.lifecycle_epoch}:{nonce}"
    )
    assert await backend.activate_lifecycle_mirror(
        identity.lifecycle_cleanup_key,
        identity.lifecycle_epoch,
        nonce,
    )
    assert await backend.set_recent_window_for_lifecycle(
        build_recent_window_key("usr_cleanup", "cnv_cleanup"),
        [{"role": "user", "text": "private"}],
        user_id="usr_cleanup",
        conversation_id="cnv_cleanup",
        lifecycle_cleanup_key=identity.lifecycle_cleanup_key,
        lifecycle_epoch=identity.lifecycle_epoch,
        cache_revision=identity.cache_revision,
        derivation_revision=identity.derivation_revision,
        conversation_lifecycle_epoch="conversation_cleanup_epoch",
        conversation_source_revision=1,
    )
    assert await backend.set_context_view_if_newer_for_lifecycle(
        "ctx-cleanup",
        {"user_id": "usr_cleanup", "conversation_id": "cnv_cleanup"},
        600,
        1,
        lifecycle_cleanup_key=identity.lifecycle_cleanup_key,
        lifecycle_epoch=identity.lifecycle_epoch,
    )
    notification_id = await backend.publish_job_notification(
        "atagia:test",
        {"job_id": "job_cleanup"},
        lifecycle_cleanup_key=identity.lifecycle_cleanup_key,
        lifecycle_epoch=identity.lifecycle_epoch,
    )
    assert notification_id is not None

    service = ConversationLifecycleService(_runtime(database_path, backend))
    with pytest.raises(UserErasureCleanupPendingError):
        await service.erase_user_data(
            connection,
            user_id="usr_cleanup",
            confirmation=ERASE_ALL_DATA_CONFIRMATION,
        )

    assert (
        await (
            await connection.execute("SELECT 1 FROM users WHERE id = 'usr_cleanup'")
        ).fetchone()
        is None
    )
    pending = await UserErasureRepository(
        connection,
        CLOCK,
    ).get_erasure_state_for_candidate("usr_cleanup")
    assert pending is not None
    assert pending["erasure_cleanup_state"] == "pending"
    assert pending["cleanup_id"] is not None
    recent_window_key = build_recent_window_key("usr_cleanup", "cnv_cleanup")
    assert await backend.get_recent_window(recent_window_key) is not None

    await close_connection(connection)
    connection = await open_connection(database_path)
    backend.fail_revoke = False
    try:
        recovery = await recover_pending_user_erasures(
            connection,
            CLOCK,
            backend,
            storage_backend_name="inprocess",
        )
        assert recovery.completed == 1
        assert recovery.pending == 0
        state = await UserErasureRepository(
            connection,
            CLOCK,
        ).get_erasure_state_for_candidate("usr_cleanup")
        assert state is not None
        assert state["erasure_cleanup_state"] == "verified"
        assert state["cleanup_id"] is None
        assert await backend.get_recent_window(recent_window_key) is None
        assert await backend.get_context_view("ctx-cleanup") is None
        assert (
            await backend.publish_job_notification(
                "atagia:test",
                {"job_id": "job_stale"},
                lifecycle_cleanup_key=identity.lifecycle_cleanup_key,
                lifecycle_epoch=identity.lifecycle_epoch,
            )
            is None
        )
        report = await service.erase_user_data(
            connection,
            user_id="usr_cleanup",
            confirmation=ERASE_ALL_DATA_CONFIRMATION,
        )
        assert report.already_erased is True
        with pytest.raises(UserDeletedError):
            await UserRepository(connection, CLOCK).create_user("usr_cleanup")
    finally:
        await close_connection(connection)


@pytest.mark.asyncio
async def test_malformed_legacy_transient_candidate_keeps_cleanup_pending(
    tmp_path: Path,
) -> None:
    database_path = str(tmp_path / "malformed-legacy-transient.db")
    backend = _MalformedLegacyBackend()
    connection = await initialize_database(database_path, MIGRATIONS_DIR)
    try:
        await UserRepository(connection, CLOCK).create_user("usr_cleanup")
        identity = await UserLifecycleRepository(
            connection,
            CLOCK,
        ).get_active_identity("usr_cleanup")
        assert identity is not None
        nonce = "prepare-malformed-legacy-cleanup"
        assert (
            await backend.prepare_lifecycle_mirror(
                identity.lifecycle_cleanup_key,
                identity.lifecycle_epoch,
                nonce,
            )
            == f"preparing:{identity.lifecycle_epoch}:{nonce}"
        )
        assert await backend.activate_lifecycle_mirror(
            identity.lifecycle_cleanup_key,
            identity.lifecycle_epoch,
            nonce,
        )

        service = ConversationLifecycleService(_runtime(database_path, backend))
        with pytest.raises(UserErasureCleanupPendingError):
            await service.erase_user_data(
                connection,
                user_id="usr_cleanup",
                confirmation=ERASE_ALL_DATA_CONFIRMATION,
            )

        repository = UserErasureRepository(connection, CLOCK)
        state = await repository.get_erasure_state_for_candidate("usr_cleanup")
        assert state is not None
        assert state["erasure_cleanup_state"] == "pending"
        cleanup_id = str(state["cleanup_id"])
        cleanup = await repository.get_cleanup(cleanup_id)
        assert cleanup is not None
        assert cleanup["attempt_count"] == 1
        assert "malformed candidates" in str(cleanup["last_error"])
        targets = await repository.list_cleanup_targets(cleanup_id)
        assert [target["checkpoint_state"] for target in targets] == ["pending"]

        backend.malformed_candidates = 0
        recovery = await recover_pending_user_erasures(
            connection,
            CLOCK,
            backend,
            storage_backend_name="inprocess",
        )
        assert recovery.completed == 1
        assert recovery.pending == 0
    finally:
        await close_connection(connection)


@pytest.mark.asyncio
async def test_legacy_transient_purge_holds_sqlite_cleanup_fence(
    tmp_path: Path,
) -> None:
    database_path = str(tmp_path / "legacy-purge-sqlite-fence.db")
    backend = _PausingLegacyPurgeBackend()
    first_connection = await initialize_database(database_path, MIGRATIONS_DIR)
    second_connection = None
    first_task: asyncio.Task[dict[str, object] | None] | None = None
    second_task: asyncio.Task[dict[str, object] | None] | None = None
    try:
        await UserRepository(first_connection, CLOCK).create_user("usr_cleanup")
        identity = await UserLifecycleRepository(
            first_connection,
            CLOCK,
        ).get_active_identity("usr_cleanup")
        assert identity is not None
        nonce = "prepare-legacy-purge-fence"
        assert (
            await backend.prepare_lifecycle_mirror(
                identity.lifecycle_cleanup_key,
                identity.lifecycle_epoch,
                nonce,
            )
            == f"preparing:{identity.lifecycle_epoch}:{nonce}"
        )
        assert await backend.activate_lifecycle_mirror(
            identity.lifecycle_cleanup_key,
            identity.lifecycle_epoch,
            nonce,
        )

        lifecycle_service = ConversationLifecycleService(
            _runtime(database_path, backend)
        )
        with pytest.raises(UserErasureCleanupPendingError):
            await lifecycle_service.erase_user_data(
                first_connection,
                user_id="usr_cleanup",
                confirmation=ERASE_ALL_DATA_CONFIRMATION,
            )
        state = await UserErasureRepository(
            first_connection,
            CLOCK,
        ).get_erasure_state_for_candidate("usr_cleanup")
        assert state is not None
        cleanup_id = str(state["cleanup_id"])

        backend.fail_revoke = False
        backend.pause_next_purge = True
        first_task = asyncio.create_task(
            UserErasureCleanupService(
                first_connection,
                CLOCK,
                backend,
                "inprocess",
            ).complete_cleanup(cleanup_id)
        )
        await asyncio.wait_for(backend.purge_paused.wait(), timeout=2)

        second_connection = await open_connection(database_path)
        second_task = asyncio.create_task(
            UserErasureCleanupService(
                second_connection,
                CLOCK,
                backend,
                "inprocess",
            ).complete_cleanup(cleanup_id)
        )
        await asyncio.sleep(0.05)
        assert not second_task.done()

        backend.resume_purge.set()
        results = await asyncio.wait_for(
            asyncio.gather(first_task, second_task),
            timeout=3,
        )
        verified = [
            result
            for result in results
            if result is not None and result.get("erasure_cleanup_state") == "verified"
        ]
        assert len(verified) == 1
        assert 1 <= backend.legacy_purge_calls <= 2
    finally:
        backend.resume_purge.set()
        pending_tasks = [
            task
            for task in (first_task, second_task)
            if task is not None and not task.done()
        ]
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)
        if second_connection is not None:
            await close_connection(second_connection)
        await close_connection(first_connection)


@pytest.mark.asyncio
async def test_stale_cleaner_cannot_purge_recreated_user_lifecycle(
    tmp_path: Path,
) -> None:
    database_path = str(tmp_path / "stale-cleaner-recreation.db")
    backend = _PausingLifecycleBackend()
    first_connection = await initialize_database(database_path, MIGRATIONS_DIR)
    second_connection = None
    stale_cleanup_task: asyncio.Task[dict[str, object] | None] | None = None
    try:
        await UserRepository(first_connection, CLOCK).create_user("usr_cleanup")
        original_identity = await UserLifecycleRepository(
            first_connection,
            CLOCK,
        ).get_active_identity("usr_cleanup")
        assert original_identity is not None
        original_nonce = "prepare-original-lifecycle"
        assert (
            await backend.prepare_lifecycle_mirror(
                original_identity.lifecycle_cleanup_key,
                original_identity.lifecycle_epoch,
                original_nonce,
            )
            == f"preparing:{original_identity.lifecycle_epoch}:{original_nonce}"
        )
        assert await backend.activate_lifecycle_mirror(
            original_identity.lifecycle_cleanup_key,
            original_identity.lifecycle_epoch,
            original_nonce,
        )

        lifecycle_service = ConversationLifecycleService(
            _runtime(database_path, backend)
        )
        with pytest.raises(UserErasureCleanupPendingError):
            await lifecycle_service.erase_user_data(
                first_connection,
                user_id="usr_cleanup",
                confirmation=ERASE_ALL_DATA_CONFIRMATION,
            )
        pending = await UserErasureRepository(
            first_connection,
            CLOCK,
        ).get_erasure_state_for_candidate("usr_cleanup")
        assert pending is not None
        cleanup_id = str(pending["cleanup_id"])

        backend.fail_revoke = False
        backend.pause_next_revoke = True
        stale_cleaner = UserErasureCleanupService(
            first_connection,
            CLOCK,
            backend,
            "inprocess",
        )
        stale_cleanup_task = asyncio.create_task(
            stale_cleaner.complete_cleanup(cleanup_id)
        )
        await asyncio.wait_for(backend.revoke_paused.wait(), timeout=2)

        second_connection = await open_connection(database_path)
        winning_cleaner = UserErasureCleanupService(
            second_connection,
            CLOCK,
            backend,
            "inprocess",
        )
        verified = await winning_cleaner.complete_cleanup(cleanup_id)
        assert verified is not None
        assert verified["erasure_cleanup_state"] == "verified"

        repository = UserErasureRepository(second_connection, CLOCK)
        complete = await repository.get_erasure_state_for_candidate("usr_cleanup")
        assert complete is not None
        assert await repository.retire_tombstone(
            str(complete["tombstone_id"]),
            expected_row_version=int(complete["erasure_row_version"]),
            deleted_before="2026-08-01T00:00:00+00:00",
        )
        await UserRepository(second_connection, CLOCK).create_user("usr_cleanup")
        recreated_identity = await UserLifecycleRepository(
            second_connection,
            CLOCK,
        ).get_active_identity("usr_cleanup")
        assert recreated_identity is not None
        assert recreated_identity.lifecycle_epoch != original_identity.lifecycle_epoch
        assert (
            recreated_identity.lifecycle_cleanup_key
            != original_identity.lifecycle_cleanup_key
        )

        recreated_nonce = "prepare-recreated-lifecycle"
        assert (
            await backend.prepare_lifecycle_mirror(
                recreated_identity.lifecycle_cleanup_key,
                recreated_identity.lifecycle_epoch,
                recreated_nonce,
            )
            == f"preparing:{recreated_identity.lifecycle_epoch}:{recreated_nonce}"
        )
        assert await backend.activate_lifecycle_mirror(
            recreated_identity.lifecycle_cleanup_key,
            recreated_identity.lifecycle_epoch,
            recreated_nonce,
        )
        assert await backend.set_context_view_if_newer_for_lifecycle(
            "ctx-recreated",
            {"user_id": "usr_cleanup", "conversation_id": "cnv_recreated"},
            600,
            1,
            lifecycle_cleanup_key=recreated_identity.lifecycle_cleanup_key,
            lifecycle_epoch=recreated_identity.lifecycle_epoch,
        )
        recent_window_key = build_recent_window_key(
            "usr_cleanup",
            "cnv_recreated",
        )
        assert await backend.set_recent_window_for_lifecycle(
            recent_window_key,
            [{"role": "user", "text": "new lifecycle"}],
            user_id="usr_cleanup",
            conversation_id="cnv_recreated",
            lifecycle_cleanup_key=recreated_identity.lifecycle_cleanup_key,
            lifecycle_epoch=recreated_identity.lifecycle_epoch,
            cache_revision=recreated_identity.cache_revision,
            derivation_revision=recreated_identity.derivation_revision,
            conversation_lifecycle_epoch="cle_recreated",
            conversation_source_revision=1,
        )
        new_notification_id = await backend.publish_job_notification(
            "atagia:recreated",
            {"job_id": "job_recreated", "user_id": "usr_cleanup"},
            lifecycle_cleanup_key=recreated_identity.lifecycle_cleanup_key,
            lifecycle_epoch=recreated_identity.lifecycle_epoch,
        )
        assert new_notification_id is not None

        backend.resume_revoke.set()
        assert await asyncio.wait_for(stale_cleanup_task, timeout=2) is None
        assert backend.legacy_purge_calls == 1

        assert await backend.get_context_view("ctx-recreated") == {
            "user_id": "usr_cleanup",
            "conversation_id": "cnv_recreated",
        }
        assert await backend.get_recent_window(recent_window_key) == [
            {"role": "user", "text": "new lifecycle"}
        ]
        notification = await backend.dequeue_job(
            "stream:atagia:recreated",
            timeout_seconds=0,
        )
        assert notification == {
            "message_id": new_notification_id,
            "payload": {"job_id": "job_recreated", "user_id": "usr_cleanup"},
        }
    finally:
        backend.resume_revoke.set()
        if stale_cleanup_task is not None and not stale_cleanup_task.done():
            await asyncio.gather(stale_cleanup_task, return_exceptions=True)
        if second_connection is not None:
            await close_connection(second_connection)
        await close_connection(first_connection)
