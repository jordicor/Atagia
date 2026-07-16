"""Integration tests for prompt-time prepared initial-context package reads."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from atagia.app import AppRuntime, initialize_runtime
from atagia.core.config import Settings
from atagia.core.initial_context_package_repository import (
    InitialContextPackageRepository,
)
from atagia.core.retrieval_event_repository import RetrievalEventRepository
from atagia.core.repositories import ConversationRepository
from atagia.core.space_repository import SpaceRepository
from atagia.models.schemas_initial_context_package import InitialContextPackageKind
from atagia.models.schemas_replay import AblationConfig
from atagia.services.chat_service import ChatService
from atagia.services.chat_support import (
    apply_conversation_policy_overlay,
    resolve_operational_profile,
    resolve_policy,
)
from atagia.services.initial_context_package_builder import InitialContextPackageBuilder
from atagia.services.initial_context_package_prompt import (
    _expected_packages,
    _read_expected_package,
)
from atagia.services.initial_context_package_refresh_service import (
    InitialContextPackageRefreshEnqueuer,
)
from atagia.services.job_tracking_service import JobTrackingService
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)
from atagia.services.sidecar_service import SidecarService
from atagia.models.schemas_jobs import InitialContextPackageRefreshReason
from atagia.models.schemas_jobs import WorkerControlMode
from atagia.models.schemas_memory import SpaceBoundaryMode
from atagia.services.worker_control_service import WorkerControlService

MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "migrations"
)
MANIFESTS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "manifests"
)
_CANDIDATE_SCORE_KEY_PATTERN = re.compile(
    r'<candidate[^>]*memory_id="([^"]+)"[^>]*score_key="([^"]+)"'
)


def _is_need_detection_card_purpose(purpose: object) -> bool:
    value = str(purpose)
    return value.startswith("need_detection_") and value.endswith("_card")


class PromptPackageProvider(LLMProvider):
    name = "initial-context-package-prompt-tests"

    def __init__(self) -> None:
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        purpose = str(request.metadata.get("purpose"))
        if _is_need_detection_card_purpose(purpose):
            outputs = {
                "need_detection_needs_card": "none",
                "need_detection_language_card": "en\nen",
                "need_detection_memory_card": "mixed",
                "need_detection_exact_card": "no",
                "need_detection_shape_card": "default",
                "need_detection_facets_card": "none",
                "need_detection_callback_card": "no",
                "need_detection_search_words_card": "prepared context",
                "need_detection_search_words_other_language_card": "none",
            }
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=outputs[purpose],
            )
        if purpose == "context_cache_signal_detection":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps(
                    {
                        "contradiction_detected": False,
                        "high_stakes_topic": False,
                        "sensitive_content": False,
                        "mode_shift_target": None,
                        "short_followup": True,
                        "ambiguous_wording": False,
                    }
                ),
            )
        if purpose == "applicability_relevance_card":
            candidate_keys = _CANDIDATE_SCORE_KEY_PATTERN.findall(
                request.messages[1].content
            )
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text="\n".join(
                    f"{score_key} useful" for _memory_id, score_key in candidate_keys
                ),
            )
        if purpose == "applicability_date_card":
            candidate_keys = _CANDIDATE_SCORE_KEY_PATTERN.findall(
                request.messages[1].content
            )
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text="\n".join(
                    f"{score_key} none" for _memory_id, score_key in candidate_keys
                ),
            )
        if purpose == "consent_confirmation_intent":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=json.dumps({"intent": "ambiguous"}),
            )
        if purpose == "chat_reply":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text="Prepared context was available.",
            )
        raise AssertionError(f"Unexpected purpose: {purpose}")

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError(f"Embeddings are not used in this test: {request.model}")


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        sqlite_path=str(tmp_path / "atagia-initial-context-package-prompt.db"),
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
) -> tuple[AppRuntime, PromptPackageProvider]:
    provider = PromptPackageProvider()
    monkeypatch.setattr(
        "atagia.app.build_llm_client",
        lambda _settings: LLMClient(provider_name=provider.name, providers=[provider]),
    )
    runtime = await initialize_runtime(_settings(tmp_path))
    return runtime, provider


async def _warm_conversation(runtime: AppRuntime) -> None:
    sidecar = SidecarService(runtime)
    await sidecar.get_context(
        user_id="usr_1",
        conversation_id="cnv_1",
        message="Warm this conversation before package materialization.",
        mode="coding_debug",
        platform_id="aurvek",
        message_id="warm-user-1",
    )
    await sidecar.add_response(
        user_id="usr_1",
        conversation_id="cnv_1",
        text="Conversation warmed.",
        platform_id="aurvek",
        message_id="warm-assistant-1",
    )


async def _materialize_packages(
    runtime: AppRuntime,
    *,
    privacy_enforcement: str = "enforce",
) -> None:
    connection = await runtime.open_connection()
    try:
        conversation = await ConversationRepository(
            connection,
            runtime.clock,
        ).get_conversation("cnv_1", "usr_1")
        assert conversation is not None
        operational_profile = resolve_operational_profile(
            loader=runtime.operational_profile_loader,
            settings=runtime.settings,
        )
        resolved_policy = resolve_policy(
            runtime.manifests,
            "coding_debug",
            runtime.policy_resolver,
            operational_profile,
        )
        resolved_policy = apply_conversation_policy_overlay(
            resolved_policy,
            conversation,
        )
        builder = InitialContextPackageBuilder(connection, runtime.clock)
        workspace_id = conversation.get("workspace_id")
        character_id = conversation.get("character_id") or workspace_id
        await builder.build_baseline_package(
            user_id="usr_1",
            resolved_policy=resolved_policy,
            workspace_id=workspace_id,
            assistant_mode_id="coding_debug",
            user_persona_id=conversation.get("user_persona_id"),
            platform_id=conversation.get("platform_id"),
            character_id=character_id,
            active_presence_id=conversation.get("active_presence_id"),
            active_space_id=conversation.get("active_space_id"),
            active_mind_id=conversation.get("active_mind_id"),
            mind_topology=conversation.get("mind_topology"),
            active_embodiment_id=conversation.get("active_embodiment_id"),
            active_realm_id=conversation.get("active_realm_id"),
            incognito=bool(conversation.get("incognito"))
            or bool(conversation.get("isolated_mode")),
            privacy_enforcement=privacy_enforcement,
            operational_profile=operational_profile.snapshot,
        )
        await builder.build_conversation_package(
            user_id="usr_1",
            conversation_id="cnv_1",
            conversation=conversation,
            resolved_policy=resolved_policy,
            privacy_enforcement=privacy_enforcement,
            operational_profile=operational_profile.snapshot,
        )
    finally:
        await connection.close()


async def _stale_packages(runtime: AppRuntime) -> None:
    connection = await runtime.open_connection()
    try:
        repository = InitialContextPackageRepository(connection, runtime.clock)
        await repository.mark_stale_for_key_family(
            user_id="usr_1",
            package_kind=InitialContextPackageKind.BASELINE,
            retrieval_profile_id="coding_debug",
        )
        await repository.mark_stale_for_key_family(
            user_id="usr_1",
            package_kind=InitialContextPackageKind.CONVERSATION,
            retrieval_profile_id="coding_debug",
            conversation_id="cnv_1",
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_sidecar_reads_prepared_package_without_llm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, provider = await _build_runtime(tmp_path, monkeypatch)
    try:
        await _warm_conversation(runtime)
        provider.requests.clear()
        await _materialize_packages(runtime)

        context = await SidecarService(runtime).get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="Use the prepared package.",
            mode="coding_debug",
            platform_id="aurvek",
            message_id="turn-user-2",
            ablation=AblationConfig(
                disable_context_cache=True,
                skip_need_detection=True,
                skip_applicability_scoring=True,
            ),
        )

        assert provider.requests == []
        assert context.initial_context_package["rendered"] is True
        assert {
            package["package_kind"]: package["status"]
            for package in context.initial_context_package["packages"]
        } == {"baseline": "hit", "conversation": "stale"}
        assert context.initial_context_package["refresh_requested"] is True
        assert (
            context.initial_context_package["refresh_jobs"]["conversation"] is not None
        )
        assert "<prepared_initial_context>" in context.system_prompt
        assert "Baseline Prepared Context" in context.system_prompt
        assert "Conversation Prepared Context" not in context.system_prompt

        connection = await runtime.open_connection()
        try:
            row = await (
                await connection.execute(
                    """
                    SELECT status, recovery_envelope_json
                    FROM worker_job_runs
                    WHERE job_id = ?
                    """,
                    (context.initial_context_package["refresh_jobs"]["conversation"],),
                )
            ).fetchone()
            assert row is not None
            assert row["status"] == "queued"
            recovery_envelope = json.loads(row["recovery_envelope_json"])
            assert recovery_envelope["payload"]["package_kind"] == "conversation"
            assert recovery_envelope["payload"]["reason"] == "source_changed"
        finally:
            await connection.close()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_sidecar_reports_stale_and_signature_mismatch_without_rendering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mismatch_path = tmp_path / "mismatch"
    mismatch_path.mkdir()
    runtime, _provider = await _build_runtime(mismatch_path, monkeypatch)
    try:
        await _warm_conversation(runtime)
        await _materialize_packages(runtime)
        await _stale_packages(runtime)

        stale_context = await SidecarService(runtime).get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="Do not render stale package.",
            mode="coding_debug",
            platform_id="aurvek",
            message_id="turn-user-stale",
        )

        assert stale_context.initial_context_package["rendered"] is False
        assert {
            package["status"]
            for package in stale_context.initial_context_package["packages"]
        } == {"stale"}
        assert stale_context.initial_context_package["refresh_requested"] is True
        assert "<prepared_initial_context>" not in stale_context.system_prompt
    finally:
        await runtime.close()

    runtime, _provider = await _build_runtime(tmp_path, monkeypatch)
    try:
        await _warm_conversation(runtime)
        await _materialize_packages(runtime, privacy_enforcement="off")

        mismatch_context = await SidecarService(runtime).get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="Do not render mismatched package.",
            mode="coding_debug",
            platform_id="aurvek",
            message_id="turn-user-mismatch",
        )

        assert mismatch_context.initial_context_package["rendered"] is False
        assert {
            package["status"]
            for package in mismatch_context.initial_context_package["packages"]
        } == {"miss"}
        assert mismatch_context.initial_context_package["refresh_requested"] is True
        assert "<prepared_initial_context>" not in mismatch_context.system_prompt
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_coordinate_key_miss_does_not_stale_current_baseline_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider = await _build_runtime(tmp_path, monkeypatch)
    try:
        await _warm_conversation(runtime)
        setup_connection = await runtime.open_connection()
        try:
            await SpaceRepository(setup_connection, runtime.clock).resolve_space(
                owner_user_id="usr_1",
                space_id="space_variant",
                boundary_mode=SpaceBoundaryMode.FOCUS,
                display_name="Coordinate variant",
                source_kind="explicit",
                source_id="space_variant",
            )
        finally:
            await setup_connection.close()
        await _materialize_packages(runtime)
        connection = await runtime.open_connection()
        try:
            repository = InitialContextPackageRepository(connection, runtime.clock)
            baseline_row = await (
                await connection.execute(
                    """
                    SELECT package_key_hash
                    FROM initial_context_packages
                    WHERE user_id = ?
                      AND package_kind = 'baseline'
                      AND build_status = 'active'
                    LIMIT 1
                    """,
                    ("usr_1",),
                )
            ).fetchone()
            assert baseline_row is not None
            original = await repository.get_by_key_hash(
                user_id="usr_1",
                package_key_hash=str(baseline_row["package_key_hash"]),
                include_inactive=True,
            )
            assert original is not None
            original_bytes = original.model_dump(mode="json")
            source_conversation = await ConversationRepository(
                connection,
                runtime.clock,
            ).get_conversation("cnv_1", "usr_1")
            assert source_conversation is not None
            operational_profile = resolve_operational_profile(
                loader=runtime.operational_profile_loader,
                settings=runtime.settings,
            )
            canonical_policy = resolve_policy(
                runtime.manifests,
                "coding_debug",
                runtime.policy_resolver,
                operational_profile,
            )
            canonical_policy = apply_conversation_policy_overlay(
                canonical_policy,
                source_conversation,
            )
            stale_caller = dict(source_conversation)
            stale_caller["incognito"] = 1
            stale_caller["isolated_mode"] = 1
            canonical_expected = await _expected_packages(
                connection,
                clock=runtime.clock,
                user_id="usr_1",
                conversation_id="cnv_1",
                conversation=stale_caller,
                resolved_policy=canonical_policy,
                privacy_enforcement="enforce",
                operational_profile=operational_profile.snapshot,
            )
            canonical_reads = [
                await _read_expected_package(repository, request)
                for request in canonical_expected
            ]
            assert len(canonical_reads) == 2
            assert all(read.package is not None for read in canonical_reads)
            assert {read.diagnostics["status"] for read in canonical_reads} == {"hit"}

            await ConversationRepository(connection, runtime.clock).create_conversation(
                "cnv_variant",
                "usr_1",
                source_conversation.get("workspace_id"),
                "coding_debug",
                "Coordinate variant",
                user_persona_id=original.key_json.subject_json["user_persona_id"],
                platform_id=original.key_json.subject_json["platform_id"],
                character_id=original.key_json.subject_json["character_id"],
                active_space_id="space_variant",
            )
            conversation = await ConversationRepository(
                connection,
                runtime.clock,
            ).get_conversation("cnv_variant", "usr_1")
            assert conversation is not None
            resolved_policy = resolve_policy(
                runtime.manifests,
                "coding_debug",
                runtime.policy_resolver,
                operational_profile,
            )
            resolved_policy = apply_conversation_policy_overlay(
                resolved_policy,
                conversation,
            )
            expected = await _expected_packages(
                connection,
                clock=runtime.clock,
                user_id="usr_1",
                conversation_id="cnv_variant",
                conversation=conversation,
                resolved_policy=resolved_policy,
                privacy_enforcement="enforce",
                operational_profile=operational_profile.snapshot,
            )
            baseline_request = next(
                request
                for request in expected
                if request.kind == InitialContextPackageKind.BASELINE
            )
            read = await _read_expected_package(repository, baseline_request)

            assert read.package is None
            assert read.diagnostics["status"] == "signature_mismatch"
            assert read.refresh_required is True
            after = await repository.get_by_key_hash(
                user_id="usr_1",
                package_key_hash=original.package_key_hash,
                include_inactive=True,
            )
            assert after is not None
            assert after.model_dump(mode="json") == original_bytes
        finally:
            await connection.close()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_chat_records_prepared_package_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, provider = await _build_runtime(tmp_path, monkeypatch)
    try:
        await _warm_conversation(runtime)
        await _materialize_packages(runtime)

        result = await ChatService(runtime).chat_reply(
            user_id="usr_1",
            conversation_id="cnv_1",
            message_text="Answer with prepared context available.",
            assistant_mode_id="coding_debug",
            platform_id="aurvek",
            debug=True,
        )

        chat_request = [
            request
            for request in provider.requests
            if request.metadata.get("purpose") == "chat_reply"
        ][-1]
        assert "<prepared_initial_context>" in chat_request.messages[0].content
        assert result.debug is not None
        assert result.debug["initial_context_package"]["rendered"] is True
        connection = await runtime.open_connection()
        try:
            event = await RetrievalEventRepository(
                connection,
                runtime.clock,
            ).get_event(result.retrieval_event_id, "usr_1")
            assert event is not None
            assert event["outcome_json"]["initial_context_package"]["rendered"] is True
        finally:
            await connection.close()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_refresh_enqueue_marks_existing_package_stale_before_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider = await _build_runtime(tmp_path, monkeypatch)
    try:
        await _warm_conversation(runtime)
        await _materialize_packages(runtime)

        connection = await runtime.open_connection()
        try:
            repository = InitialContextPackageRepository(connection, runtime.clock)
            before = await repository.get_latest_for_conversation(
                user_id="usr_1",
                conversation_id="cnv_1",
                retrieval_profile_id="coding_debug",
            )
            assert before is not None
            enqueuer = InitialContextPackageRefreshEnqueuer(
                storage_backend=runtime.storage_backend,
                clock=runtime.clock,
                job_tracking_service=JobTrackingService(
                    connection,
                    runtime.clock,
                    workers_enabled=runtime.settings.workers_enabled,
                    settings=runtime.settings,
                ),
                package_repository=repository,
            )
            await enqueuer.enqueue_refresh(
                user_id="usr_1",
                conversation_id="cnv_1",
                retrieval_profile_id="coding_debug",
                reason=InitialContextPackageRefreshReason.COORDINATE_CHANGE,
                source_message_ids=["warm-assistant-1"],
                operational_profile=resolve_operational_profile(
                    loader=runtime.operational_profile_loader,
                    settings=runtime.settings,
                ).snapshot,
            )
            stale = await repository.read_by_key_hash(
                user_id="usr_1",
                package_key_hash=before.package_key_hash,
            )
            assert stale.status == "stale"
        finally:
            await connection.close()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_refresh_disabled_still_marks_existing_package_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider = await _build_runtime(tmp_path, monkeypatch)
    try:
        await _warm_conversation(runtime)
        await _materialize_packages(runtime)

        connection = await runtime.open_connection()
        try:
            repository = InitialContextPackageRepository(connection, runtime.clock)
            before = await repository.get_latest_for_conversation(
                user_id="usr_1",
                conversation_id="cnv_1",
                retrieval_profile_id="coding_debug",
            )
            assert before is not None
            enqueuer = InitialContextPackageRefreshEnqueuer(
                storage_backend=runtime.storage_backend,
                clock=runtime.clock,
                package_repository=repository,
                refresh_enabled=False,
            )
            refresh_job_id = await enqueuer.enqueue_refresh(
                user_id="usr_1",
                conversation_id="cnv_1",
                retrieval_profile_id="coding_debug",
                reason=InitialContextPackageRefreshReason.MESSAGE_WRITE,
                source_message_ids=["warm-assistant-1"],
                operational_profile=resolve_operational_profile(
                    loader=runtime.operational_profile_loader,
                    settings=runtime.settings,
                ).snapshot,
            )
            stale = await repository.read_by_key_hash(
                user_id="usr_1",
                package_key_hash=before.package_key_hash,
            )
            assert refresh_job_id is None
            assert stale.status == "stale"
        finally:
            await connection.close()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_read_side_refresh_does_not_stale_newer_active_row_when_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider = await _build_runtime(tmp_path, monkeypatch)
    try:
        await _warm_conversation(runtime)
        await _materialize_packages(runtime)

        connection = await runtime.open_connection()
        try:
            repository = InitialContextPackageRepository(connection, runtime.clock)
            observed = await repository.get_latest_for_conversation(
                user_id="usr_1",
                conversation_id="cnv_1",
                retrieval_profile_id="coding_debug",
            )
            assert observed is not None

            await _materialize_packages(runtime)
            newer = await repository.get_latest_for_conversation(
                user_id="usr_1",
                conversation_id="cnv_1",
                retrieval_profile_id="coding_debug",
            )
            assert newer is not None
            assert newer.package_row_version > observed.package_row_version
            newer_bytes = newer.model_dump(mode="json")

            refresh_job_id = await InitialContextPackageRefreshEnqueuer(
                storage_backend=runtime.storage_backend,
                clock=runtime.clock,
                package_repository=repository,
                refresh_enabled=False,
            ).enqueue_refresh(
                user_id="usr_1",
                conversation_id="cnv_1",
                package_kind=InitialContextPackageKind.CONVERSATION,
                retrieval_profile_id="coding_debug",
                reason=InitialContextPackageRefreshReason.SOURCE_CHANGED,
                mark_existing_packages_stale=False,
            )

            after = await repository.get_latest_for_conversation(
                user_id="usr_1",
                conversation_id="cnv_1",
                retrieval_profile_id="coding_debug",
                include_inactive=True,
            )
            assert refresh_job_id is None
            assert after is not None
            assert after.model_dump(mode="json") == newer_bytes
        finally:
            await connection.close()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_paused_worker_control_stales_package_without_enqueuing_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _provider = await _build_runtime(tmp_path, monkeypatch)
    try:
        await _warm_conversation(runtime)
        await _materialize_packages(runtime)

        connection = await runtime.open_connection()
        try:
            repository = InitialContextPackageRepository(connection, runtime.clock)
            before = await repository.get_latest_for_conversation(
                user_id="usr_1",
                conversation_id="cnv_1",
                retrieval_profile_id="coding_debug",
            )
            assert before is not None
            await WorkerControlService(connection, runtime.clock).set_mode(
                WorkerControlMode.PAUSE_NEW_JOBS,
                reason="rollout test pause",
            )
        finally:
            await connection.close()

        write_result = await SidecarService(runtime).add_response(
            user_id="usr_1",
            conversation_id="cnv_1",
            text="Persist this while workers are paused.",
            platform_id="aurvek",
            message_id="paused-worker-assistant-1",
        )
        assert write_result.created is True

        connection = await runtime.open_connection()
        try:
            stale = await InitialContextPackageRepository(
                connection,
                runtime.clock,
            ).read_by_key_hash(
                user_id="usr_1",
                package_key_hash=before.package_key_hash,
            )
            assert stale.status == "stale"
        finally:
            await connection.close()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_sidecar_drops_prepared_package_before_live_context_under_overflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, provider = await _build_runtime(tmp_path, monkeypatch)
    try:
        await _warm_conversation(runtime)
        provider.requests.clear()
        await _materialize_packages(runtime)

        context = await SidecarService(runtime).get_context(
            user_id="usr_1",
            conversation_id="cnv_1",
            message="Use live context first when budget is tight.",
            mode="coding_debug",
            platform_id="aurvek",
            message_id="turn-user-overflow",
            ablation=AblationConfig(
                disable_context_cache=True,
                skip_need_detection=True,
                skip_applicability_scoring=True,
                context_envelope_budget_tokens=1000,
            ),
        )

        assert provider.requests == []
        assert context.initial_context_package["overflow_dropped"] is True
        assert context.initial_context_package["rendered"] is False
        assert "<prepared_initial_context>" not in context.system_prompt
    finally:
        await runtime.close()
