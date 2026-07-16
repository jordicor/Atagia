"""Tests for sidecar request/response memory orchestration."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from atagia.app import AppRuntime, initialize_runtime
from atagia.core.config import Settings
from atagia.core.initial_context_package_repository import (
    InitialContextPackageRepository,
)
from atagia.core.repositories import ConversationRepository, UserRepository
from atagia.core.space_repository import SpaceRepository
from atagia.models.schemas_memory import ConversationStatus, SpaceBoundaryMode
from atagia.models.schemas_initial_context_package import (
    InitialContextPackageBlocks,
    InitialContextPackageKey,
    InitialContextPackageKind,
)
from atagia.models.schemas_jobs import JobType
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)
from atagia.services.errors import (
    ConversationNotActiveError,
    MessageIdConflictError,
    SourceSequenceConflictError,
)
from atagia.services.sidecar_service import SidecarService

MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "migrations"
)
MANIFESTS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "manifests"
)


class NoLLMProvider(LLMProvider):
    name = "sidecar-no-llm-tests"

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        raise AssertionError(f"Sidecar test should not call LLM: {request.metadata}")

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError(f"Sidecar test should not embed: {request.model}")


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        sqlite_path=str(tmp_path / "atagia-sidecar-service.db"),
        migrations_path=str(MIGRATIONS_DIR),
        manifests_path=str(MANIFESTS_DIR),
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        openai_api_key="test-openai-key",
        openrouter_api_key=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
        llm_chat_model="reply-test-model",
        llm_forced_global_model="openai/reply-test-model",
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
        allow_insecure_http=True,
        small_corpus_token_threshold_ratio=0.0,
    )


async def _build_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AppRuntime:
    provider = NoLLMProvider()
    monkeypatch.setattr(
        "atagia.app.build_llm_client",
        lambda _settings: LLMClient(provider_name=provider.name, providers=[provider]),
    )
    return await initialize_runtime(_settings(tmp_path))


async def _job_counts(runtime: AppRuntime) -> dict[str, int]:
    connection = await runtime.open_connection()
    try:
        cursor = await connection.execute(
            """
            SELECT job_type, COUNT(*) AS count
            FROM worker_job_runs
            GROUP BY job_type
            """
        )
        return {
            str(row["job_type"]): int(row["count"]) for row in await cursor.fetchall()
        }
    finally:
        await connection.close()


async def _upsert_conversation_package(runtime: AppRuntime) -> str:
    connection = await runtime.open_connection()
    try:
        key = InitialContextPackageKey(
            version=2,
            package_kind=InitialContextPackageKind.CONVERSATION,
            user_id="usr_1",
            conversation_id="cnv_1",
            retrieval_profile_id="coding_debug",
        )
        package = await InitialContextPackageRepository(
            connection,
            runtime.clock,
        ).upsert_package(
            package_kind=key.package_kind,
            version=key.version,
            user_id=key.user_id,
            conversation_id=key.conversation_id,
            retrieval_profile_id=key.retrieval_profile_id,
            key_json=key,
            blocks_json=InitialContextPackageBlocks(
                conversation_summary_block="Prepared context to invalidate.",
            ),
        )
        return package.package_key_hash
    finally:
        await connection.close()


async def _package_status(runtime: AppRuntime, package_key_hash: str) -> str:
    connection = await runtime.open_connection()
    try:
        result = await InitialContextPackageRepository(
            connection,
            runtime.clock,
        ).read_by_key_hash(
            user_id="usr_1",
            package_key_hash=package_key_hash,
        )
        return result.status
    finally:
        await connection.close()


async def _seed_active_conversation(
    runtime: AppRuntime,
    sidecar: SidecarService,
) -> None:
    connection = await runtime.open_connection()
    try:
        await sidecar.ensure_user_exists(connection, "usr_write_race")
        await sidecar.ensure_conversation(
            connection,
            user_id="usr_write_race",
            conversation_id="cnv_write_race",
            workspace_id=None,
            assistant_mode_id="coding_debug",
        )
    finally:
        await connection.close()


async def _run_racing_sidecar_write(
    sidecar: SidecarService,
    write_path: str,
    *,
    message_id: str,
    text: str,
    source_seq: int,
) -> Any:
    if write_path == "context":
        return await sidecar.get_context(
            user_id="usr_write_race",
            conversation_id="cnv_write_race",
            message=text,
            mode="coding_debug",
            message_id=message_id,
            source_seq=source_seq,
        )
    if write_path == "ingest":
        return await sidecar.ingest_message(
            user_id="usr_write_race",
            conversation_id="cnv_write_race",
            role="user",
            text=text,
            mode="coding_debug",
            message_id=message_id,
            source_seq=source_seq,
        )
    return await sidecar.add_response(
        user_id="usr_write_race",
        conversation_id="cnv_write_race",
        text=text,
        message_id=message_id,
        source_seq=source_seq,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("write_path", ["context", "ingest", "response"])
@pytest.mark.parametrize(
    ("scenario", "message_ids", "texts", "expected_error"),
    [
        (
            "exact_retry",
            ("msg_concurrent_retry", "msg_concurrent_retry"),
            ("same stable event", "same stable event"),
            None,
        ),
        (
            "message_id_conflict",
            ("msg_concurrent_id_conflict", "msg_concurrent_id_conflict"),
            ("first event body", "conflicting event body"),
            MessageIdConflictError,
        ),
        (
            "source_seq_conflict",
            ("msg_concurrent_seq_a", "msg_concurrent_seq_b"),
            ("first sequence owner", "second sequence owner"),
            SourceSequenceConflictError,
        ),
    ],
)
async def test_sidecar_stable_event_admission_is_atomic_across_runtimes(
    write_path: str,
    scenario: str,
    message_ids: tuple[str, str],
    texts: tuple[str, str],
    expected_error: type[Exception] | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del scenario
    runtime_a = await _build_runtime(tmp_path, monkeypatch)
    runtime_b = await _build_runtime(tmp_path, monkeypatch)
    try:
        sidecar_a = SidecarService(runtime_a)
        sidecar_b = SidecarService(runtime_b)
        await _seed_active_conversation(runtime_a, sidecar_a)

        original = SidecarService._idempotent_message_if_present
        pre_read_count = 0
        both_pre_reads_complete = asyncio.Event()

        async def pause_after_initial_absent_read(
            *args: Any,
            **kwargs: Any,
        ) -> dict[str, Any] | None:
            nonlocal pre_read_count
            result = await original(*args, **kwargs)
            if result is None and pre_read_count < 2:
                pre_read_count += 1
                if pre_read_count == 2:
                    both_pre_reads_complete.set()
                await asyncio.wait_for(both_pre_reads_complete.wait(), timeout=5.0)
            return result

        monkeypatch.setattr(
            SidecarService,
            "_idempotent_message_if_present",
            staticmethod(pause_after_initial_absent_read),
        )
        outcomes = await asyncio.gather(
            _run_racing_sidecar_write(
                sidecar_a,
                write_path,
                message_id=message_ids[0],
                text=texts[0],
                source_seq=1,
            ),
            _run_racing_sidecar_write(
                sidecar_b,
                write_path,
                message_id=message_ids[1],
                text=texts[1],
                source_seq=1,
            ),
            return_exceptions=True,
        )

        errors = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
        successes = [
            outcome for outcome in outcomes if not isinstance(outcome, BaseException)
        ]
        if expected_error is None:
            assert errors == []
            assert len(successes) == 2
            if write_path == "context":
                assert {
                    result.request_message_id for result in successes
                } == {message_ids[0]}
            else:
                assert sorted(result.created for result in successes) == [False, True]
        else:
            assert len(successes) == 1
            assert len(errors) == 1
            assert isinstance(errors[0], expected_error)

        verification = await runtime_a.open_connection()
        try:
            count_row = await (
                await verification.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM messages
                    WHERE conversation_id = ? AND seq = 1
                    """,
                    ("cnv_write_race",),
                )
            ).fetchone()
        finally:
            await verification.close()
        assert count_row is not None and int(count_row["count"]) == 1
    finally:
        await runtime_b.close()
        await runtime_a.close()


@pytest.mark.asyncio
async def test_sidecar_concurrent_first_user_creation_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_a = await _build_runtime(tmp_path, monkeypatch)
    runtime_b = await _build_runtime(tmp_path, monkeypatch)
    connection_a = await runtime_a.open_connection()
    connection_b = await runtime_b.open_connection()
    try:
        original = UserRepository.get_user
        absent_read_count = 0
        both_absent_reads_complete = asyncio.Event()

        async def pause_after_absent_user_read(
            repository: UserRepository,
            user_id: str,
        ) -> dict[str, Any] | None:
            nonlocal absent_read_count
            result = await original(repository, user_id)
            if user_id == "usr_concurrent_first" and result is None and absent_read_count < 2:
                absent_read_count += 1
                if absent_read_count == 2:
                    both_absent_reads_complete.set()
                await asyncio.wait_for(both_absent_reads_complete.wait(), timeout=5.0)
            return result

        monkeypatch.setattr(UserRepository, "get_user", pause_after_absent_user_read)
        await asyncio.gather(
            SidecarService(runtime_a).ensure_user_exists(
                connection_a,
                "usr_concurrent_first",
            ),
            SidecarService(runtime_b).ensure_user_exists(
                connection_b,
                "usr_concurrent_first",
            ),
        )

        row = await UserRepository(connection_a, runtime_a.clock).get_active_user(
            "usr_concurrent_first"
        )
        lifecycle_count = await (
            await connection_a.execute(
                "SELECT COUNT(*) AS count FROM user_lifecycles WHERE user_id = ?",
                ("usr_concurrent_first",),
            )
        ).fetchone()
        assert row is not None
        assert lifecycle_count is not None and int(lifecycle_count["count"]) == 1
    finally:
        await connection_b.close()
        await connection_a.close()
        await runtime_b.close()
        await runtime_a.close()


@pytest.mark.asyncio
async def test_sidecar_concurrent_first_conversation_creation_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_a = await _build_runtime(tmp_path, monkeypatch)
    runtime_b = await _build_runtime(tmp_path, monkeypatch)
    sidecar_a = SidecarService(runtime_a)
    sidecar_b = SidecarService(runtime_b)
    seed_connection = await runtime_a.open_connection()
    connection_a = await runtime_a.open_connection()
    connection_b = await runtime_b.open_connection()
    try:
        await sidecar_a.ensure_user_exists(seed_connection, "usr_concurrent_first")
        original = ConversationRepository.get_conversation
        absent_read_count = 0
        both_absent_reads_complete = asyncio.Event()

        async def pause_after_absent_conversation_read(
            repository: ConversationRepository,
            conversation_id: str,
            user_id: str,
        ) -> dict[str, Any] | None:
            nonlocal absent_read_count
            result = await original(repository, conversation_id, user_id)
            if (
                conversation_id == "cnv_concurrent_first"
                and result is None
                and absent_read_count < 2
            ):
                absent_read_count += 1
                if absent_read_count == 2:
                    both_absent_reads_complete.set()
                await asyncio.wait_for(both_absent_reads_complete.wait(), timeout=5.0)
            return result

        monkeypatch.setattr(
            ConversationRepository,
            "get_conversation",
            pause_after_absent_conversation_read,
        )
        identity = {
            "user_id": "usr_concurrent_first",
            "conversation_id": "cnv_concurrent_first",
            "workspace_id": None,
            "assistant_mode_id": "coding_debug",
            "user_persona_id": "persona_concurrent",
            "platform_id": "platform_concurrent",
            "character_id": "character_concurrent",
            "active_presence_id": "presence_concurrent",
            "mind_topology": "unimind",
            "embodiment_id": "embodiment_concurrent",
            "realm_id": "realm_concurrent",
            "space_id": "space_concurrent",
        }
        conversations = await asyncio.gather(
            sidecar_a.ensure_conversation(connection_a, **identity),
            sidecar_b.ensure_conversation(connection_b, **identity),
        )
        assert [str(row["id"]) for row in conversations] == [
            "cnv_concurrent_first",
            "cnv_concurrent_first",
        ]
        for field, expected in (
            ("user_persona_id", "persona_concurrent"),
            ("platform_id", "platform_concurrent"),
            ("character_id", "character_concurrent"),
            ("active_presence_id", "presence_concurrent"),
            ("active_mind_id", "default_mind"),
            ("mind_topology", "unimind"),
            ("active_embodiment_id", "embodiment_concurrent"),
            ("active_realm_id", "realm_concurrent"),
            ("active_space_id", "space_concurrent"),
        ):
            assert {row[field] for row in conversations} == {expected}
    finally:
        await connection_b.close()
        await connection_a.close()
        await seed_connection.close()
        await runtime_b.close()
        await runtime_a.close()


@pytest.mark.asyncio
async def test_sidecar_defers_user_jobs_until_response_and_dedupes_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = await _build_runtime(tmp_path, monkeypatch)
    try:
        sidecar = SidecarService(runtime)

        context = await sidecar.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="Please remember that I prefer concise answers.",
            mode="coding_debug",
            message_id="host-user-1",
        )

        assert context.request_message_id == "host-user-1"
        assert await _job_counts(runtime) == {
            JobType.REFRESH_INITIAL_CONTEXT_PACKAGE.value: 2,
        }

        await sidecar.add_response(
            user_id="usr_1",
            conversation_id="cnv_1",
            text="Got it.",
            message_id="host-assistant-1",
        )

        assert await _job_counts(runtime) == {
            JobType.EXTRACT_MEMORY_CANDIDATES.value: 2,
            JobType.PROJECT_CONTRACT.value: 1,
            JobType.REFRESH_INITIAL_CONTEXT_PACKAGE.value: 3,
        }

        await sidecar.add_response(
            user_id="usr_1",
            conversation_id="cnv_1",
            text="Got it.",
            message_id="host-assistant-1",
        )

        assert await _job_counts(runtime) == {
            JobType.EXTRACT_MEMORY_CANDIDATES.value: 2,
            JobType.PROJECT_CONTRACT.value: 1,
            JobType.REFRESH_INITIAL_CONTEXT_PACKAGE.value: 3,
        }
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("write_path", ["context", "ingest", "response"])
async def test_sidecar_message_write_rejects_namespace_change(
    write_path: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = await _build_runtime(tmp_path, monkeypatch)
    resume = asyncio.Event()
    try:
        sidecar = SidecarService(runtime)
        await _seed_active_conversation(runtime, sidecar)
        setup_connection = await runtime.open_connection()
        try:
            spaces = SpaceRepository(setup_connection, runtime.clock)
            for space_id in ("space_old", "space_new"):
                await spaces.resolve_space(
                    owner_user_id="usr_write_race",
                    space_id=space_id,
                    boundary_mode=SpaceBoundaryMode.FOCUS,
                    display_name=space_id,
                    source_kind="explicit",
                    source_id=space_id,
                )
            await setup_connection.execute(
                """
                UPDATE conversations
                SET active_space_id = ?, updated_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (
                    "space_old",
                    runtime.clock.now().isoformat(),
                    "cnv_write_race",
                    "usr_write_race",
                ),
            )
            await setup_connection.commit()
        finally:
            await setup_connection.close()

        reached = asyncio.Event()
        original_recent_messages = SidecarService._recent_messages_for_write

        async def pause_before_write(
            *args: Any,
            **kwargs: Any,
        ) -> list[dict[str, Any]]:
            result = await original_recent_messages(*args, **kwargs)
            reached.set()
            await asyncio.wait_for(resume.wait(), timeout=5.0)
            return result

        monkeypatch.setattr(
            SidecarService,
            "_recent_messages_for_write",
            staticmethod(pause_before_write),
        )
        if write_path == "context":
            write = sidecar.get_context(
                user_id="usr_write_race",
                conversation_id="cnv_write_race",
                message="Context write with stale namespace.",
                mode="coding_debug",
                message_id="msg_namespace_race",
            )
        elif write_path == "ingest":
            write = sidecar.ingest_message(
                user_id="usr_write_race",
                conversation_id="cnv_write_race",
                role="user",
                text="Ingest write with stale namespace.",
                mode="coding_debug",
                message_id="msg_namespace_race",
            )
        else:
            write = sidecar.add_response(
                user_id="usr_write_race",
                conversation_id="cnv_write_race",
                text="Response write with stale namespace.",
                message_id="msg_namespace_race",
            )
        write_task = asyncio.create_task(write)

        await asyncio.wait_for(reached.wait(), timeout=5.0)
        mutation_connection = await runtime.open_connection()
        try:
            await mutation_connection.execute(
                """
                UPDATE conversations
                SET active_space_id = ?, updated_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (
                    "space_new",
                    runtime.clock.now().isoformat(),
                    "cnv_write_race",
                    "usr_write_race",
                ),
            )
            await mutation_connection.commit()
        finally:
            await mutation_connection.close()
            resume.set()

        with pytest.raises(ConversationNotActiveError):
            await write_task

        verification = await runtime.open_connection()
        try:
            cursor = await verification.execute(
                "SELECT 1 FROM messages WHERE id = ?",
                ("msg_namespace_race",),
            )
            assert await cursor.fetchone() is None
        finally:
            await verification.close()
    finally:
        resume.set()
        await runtime.close()


@pytest.mark.asyncio
async def test_set_memory_preferences_marks_initial_context_packages_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = await _build_runtime(tmp_path, monkeypatch)
    try:
        sidecar = SidecarService(runtime)
        await sidecar.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="Prepare a conversation.",
            mode="coding_debug",
        )
        package_key_hash = await _upsert_conversation_package(runtime)

        await sidecar.set_memory_preferences(
            "usr_1",
            remember_across_chats=False,
        )

        assert await _package_status(runtime, package_key_hash) == "stale"
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_set_conversation_incognito_marks_initial_context_packages_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = await _build_runtime(tmp_path, monkeypatch)
    try:
        sidecar = SidecarService(runtime)
        await sidecar.get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="Prepare a conversation.",
            mode="coding_debug",
        )
        package_key_hash = await _upsert_conversation_package(runtime)

        await sidecar.set_conversation_incognito(
            "usr_1",
            "cnv_1",
            True,
        )

        assert await _package_status(runtime, package_key_hash) == "stale"
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_new_conversation_preserves_explicit_cross_chat_false_when_incognito_false(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = await _build_runtime(tmp_path, monkeypatch)
    connection = await runtime.open_connection()
    try:
        sidecar = SidecarService(runtime)
        await sidecar.ensure_user_exists(connection, "usr_scope")

        conversation = await sidecar.ensure_conversation(
            connection,
            user_id="usr_scope",
            conversation_id="cnv_scope",
            workspace_id=None,
            assistant_mode_id="coding_debug",
            cross_chat_memory=False,
            incognito=False,
        )

        assert conversation["isolated_mode"] == 1
        assert conversation["incognito"] == 1
    finally:
        await connection.close()
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("write_path", ["context", "ingest", "response"])
async def test_sidecar_message_write_rechecks_active_status_after_lifecycle_wins(
    write_path: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = await _build_runtime(tmp_path, monkeypatch)
    try:
        sidecar = SidecarService(runtime)
        await _seed_active_conversation(runtime, sidecar)
        reached = asyncio.Event()
        resume = asyncio.Event()
        original_recent_messages = SidecarService._recent_messages_for_write

        async def pause_before_write(
            *args: Any,
            **kwargs: Any,
        ) -> list[dict[str, Any]]:
            result = await original_recent_messages(*args, **kwargs)
            reached.set()
            await asyncio.wait_for(resume.wait(), timeout=5.0)
            return result

        monkeypatch.setattr(
            SidecarService,
            "_recent_messages_for_write",
            staticmethod(pause_before_write),
        )

        if write_path == "context":
            write = sidecar.get_context(
                user_id="usr_write_race",
                conversation_id="cnv_write_race",
                message="Context write that must lose the lifecycle race.",
                mode="coding_debug",
                message_id="msg_write_race",
            )
        elif write_path == "ingest":
            write = sidecar.ingest_message(
                user_id="usr_write_race",
                conversation_id="cnv_write_race",
                role="user",
                text="Ingest write that must lose the lifecycle race.",
                mode="coding_debug",
                message_id="msg_write_race",
            )
        else:
            write = sidecar.add_response(
                user_id="usr_write_race",
                conversation_id="cnv_write_race",
                text="Response write that must lose the lifecycle race.",
                message_id="msg_write_race",
            )
        write_task = asyncio.create_task(write)

        await asyncio.wait_for(reached.wait(), timeout=5.0)
        mutation_connection = await runtime.open_connection()
        try:
            await mutation_connection.execute(
                """
                UPDATE conversations
                SET status = ?, updated_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (
                    ConversationStatus.PENDING_DELETION.value,
                    runtime.clock.now().isoformat(),
                    "cnv_write_race",
                    "usr_write_race",
                ),
            )
            await mutation_connection.commit()
        finally:
            await mutation_connection.close()
            resume.set()

        with pytest.raises(ConversationNotActiveError):
            await write_task

        verification = await runtime.open_connection()
        try:
            cursor = await verification.execute(
                "SELECT 1 FROM messages WHERE id = ?",
                ("msg_write_race",),
            )
            assert await cursor.fetchone() is None
        finally:
            await verification.close()
    finally:
        await runtime.close()
