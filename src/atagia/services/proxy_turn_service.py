"""High-level durable proxy-turn coordination and atomic job preparation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Sequence

from atagia.core.proxy_turn_repository import (
    ProxyDurableJobInsert,
    ProxyTurnClaim,
    ProxyTurnRepository,
    ProxyTurnResponseMessage,
)
from atagia.core.repositories import (
    ConversationRepository,
    MessageRepository,
    UserRepository,
)
from atagia.models.schemas_jobs import (
    INITIAL_CONTEXT_PACKAGE_STREAM_NAME,
    InitialContextPackageRefreshReason,
    JobEnvelope,
    JobType,
)
from atagia.services.chat_support import (
    RECENT_FETCH_LIMIT,
    build_message_jobs,
    resolve_operational_profile,
)
from atagia.services.job_tracking_service import JobTrackingService
from atagia.services.initial_context_package_refresh_service import (
    initial_context_package_refresh_dedupe_key,
    prepare_initial_context_package_refresh_payload,
)
from atagia.services.prompt_authority import PromptAuthorityContext


@dataclass(frozen=True, slots=True)
class ProxyTerminalJobPlan:
    """Inputs used to build every terminal root job inside the fenced commit."""

    response_retrieval_text: str
    response_occurred_at: str
    operational_profile: str | None
    operational_signals: Any | None
    ingest_origin: Any | None
    confirmation_strategy: Any | None
    memory_privacy_mode: Any | None
    authority: PromptAuthorityContext


async def build_proxy_terminal_jobs(
    runtime: Any,
    connection: Any,
    *,
    claim: ProxyTurnClaim,
    response_retrieval_text: str,
    response_occurred_at: str,
    operational_profile: str | None,
    operational_signals: Any | None,
    ingest_origin: Any | None,
    confirmation_strategy: Any | None,
    memory_privacy_mode: Any | None,
    authority: PromptAuthorityContext,
) -> list[ProxyDurableJobInsert]:
    """Build the exact stable root-job set required by a completed proxy pair."""

    conversations = ConversationRepository(connection, runtime.clock)
    messages = MessageRepository(connection, runtime.clock)
    users = UserRepository(connection, runtime.clock)
    conversation = await conversations.get_conversation(
        claim.conversation_id,
        claim.user_id,
    )
    if conversation is None:
        raise RuntimeError("Proxy conversation disappeared before terminal commit")
    input_message = await messages.get_message_for_idempotency(claim.request_message_id)
    if input_message is None:
        raise RuntimeError("Proxy input disappeared before terminal commit")
    input_prior = await messages.get_recent_messages_before_seq(
        claim.conversation_id,
        claim.user_id,
        before_seq=int(input_message["seq"]),
        limit=RECENT_FETCH_LIMIT,
    )
    response_prior = await messages.get_recent_messages(
        claim.conversation_id,
        claim.user_id,
        limit=RECENT_FETCH_LIMIT,
    )
    memory_preferences = await users.get_memory_preferences(claim.user_id)
    resolved_profile = resolve_operational_profile(
        loader=runtime.operational_profile_loader,
        settings=runtime.settings,
        operational_profile=operational_profile,
        operational_signals=operational_signals,
    )
    common = {
        "clock": runtime.clock,
        "conversation": conversation,
        "operational_profile": resolved_profile.snapshot,
        "memory_preferences": memory_preferences,
        "ingest_origin": ingest_origin,
        "confirmation_strategy": confirmation_strategy,
        "memory_privacy_mode": memory_privacy_mode,
        "active_presence_id": input_message.get("active_presence_id"),
        "source_presence_id": input_message.get("source_presence_id"),
        "active_space_id": input_message.get("space_id"),
        "active_mind_id": input_message.get("active_mind_id"),
        "source_mind_id": input_message.get("source_mind_id"),
        "active_embodiment_id": input_message.get("active_embodiment_id"),
        "active_realm_id": input_message.get("active_realm_id"),
        "privacy_enforcement": authority.privacy_enforcement,
        "authenticated_user_privilege_level": (
            authority.authenticated_user_privilege_level
        ),
        "authenticated_user_is_atagia_master": (
            authority.authenticated_user_is_atagia_master
        ),
    }
    jobs = build_message_jobs(
        **common,
        message_id=claim.request_message_id,
        prior_messages=input_prior,
        message_text=str(input_message["text"]),
        occurred_at=(
            str(input_message["occurred_at"])
            if input_message.get("occurred_at") is not None
            else None
        ),
        role=str(input_message["role"]),
    )
    jobs.extend(
        build_message_jobs(
            **{
                **common,
                "source_presence_id": input_message.get("active_presence_id"),
                "source_mind_id": input_message.get("active_mind_id"),
            },
            message_id=claim.response_message_id,
            prior_messages=response_prior,
            message_text=response_retrieval_text,
            occurred_at=response_occurred_at,
            role="assistant",
        )
    )
    if runtime.settings.initial_context_package_refresh_enabled:
        refresh_dedupe_key = initial_context_package_refresh_dedupe_key(
            user_id=claim.user_id,
            conversation_id=claim.conversation_id,
            package_kind="all",
            retrieval_profile_id=None,
            reason=InitialContextPackageRefreshReason.MESSAGE_WRITE,
            privacy_enforcement=authority.privacy_enforcement,
            operational_profile_token=(
                resolved_profile.snapshot.token
                if resolved_profile.snapshot is not None
                else None
            ),
        )
        tracking = JobTrackingService(
            connection,
            runtime.clock,
            workers_enabled=runtime.settings.workers_enabled,
            settings=runtime.settings,
        )
        if (
            await tracking.active_icp_refresh_job_id(
                user_id=claim.user_id,
                refresh_dedupe_key=refresh_dedupe_key,
            )
            is None
        ):
            refresh_payload = await prepare_initial_context_package_refresh_payload(
                connection,
                runtime.clock,
                user_id=claim.user_id,
                conversation_id=claim.conversation_id,
                package_kind=None,
                retrieval_profile_id=None,
                reason=InitialContextPackageRefreshReason.MESSAGE_WRITE,
                source_message_ids=[
                    claim.request_message_id,
                    claim.response_message_id,
                ],
                privacy_enforcement=authority.privacy_enforcement,
                operational_profile=resolved_profile.snapshot,
            )
            jobs.append(
                (
                    INITIAL_CONTEXT_PACKAGE_STREAM_NAME,
                    JobEnvelope(
                        job_id="pending-stable-proxy-id",
                        job_type=JobType.REFRESH_INITIAL_CONTEXT_PACKAGE,
                        user_id=claim.user_id,
                        conversation_id=claim.conversation_id,
                        message_ids=[
                            claim.request_message_id,
                            claim.response_message_id,
                        ],
                        payload=refresh_payload.model_dump(mode="json"),
                        created_at=runtime.clock.now(),
                        operational_profile=resolved_profile.snapshot,
                    ),
                )
            )
    return [
        _durable_job_insert(
            runtime,
            stream_name=stream_name,
            envelope=_stable_proxy_job_envelope(
                envelope,
                pair_id=claim.pair_id,
            ),
        )
        for stream_name, envelope in jobs
    ]


async def finalize_proxy_turn(
    runtime: Any,
    *,
    claim: ProxyTurnClaim,
    response: ProxyTurnResponseMessage,
    job_plan: ProxyTerminalJobPlan,
    failpoint: Any | None = None,
) -> list[ProxyDurableJobInsert]:
    connection = await runtime.open_connection()
    built_jobs: list[ProxyDurableJobInsert] = []

    async def build_jobs() -> Sequence[ProxyDurableJobInsert]:
        jobs = await build_proxy_terminal_jobs(
            runtime,
            connection,
            claim=claim,
            response_retrieval_text=job_plan.response_retrieval_text,
            response_occurred_at=job_plan.response_occurred_at,
            operational_profile=job_plan.operational_profile,
            operational_signals=job_plan.operational_signals,
            ingest_origin=job_plan.ingest_origin,
            confirmation_strategy=job_plan.confirmation_strategy,
            memory_privacy_mode=job_plan.memory_privacy_mode,
            authority=job_plan.authority,
        )
        built_jobs.extend(jobs)
        return jobs

    try:
        await ProxyTurnRepository(connection, runtime.clock).finalize(
            claim,
            response=response,
            durable_job_builder=build_jobs,
            failpoint=failpoint,
        )
        return built_jobs
    finally:
        await connection.close()


async def dispatch_proxy_terminal_jobs(
    runtime: Any,
    durable_jobs: Sequence[ProxyDurableJobInsert],
) -> None:
    """Expose committed jobs as transient wake-ups; SQLite remains recoverable."""

    if not durable_jobs:
        return
    connection = await runtime.open_connection()
    try:
        tracking = JobTrackingService(
            connection,
            runtime.clock,
            workers_enabled=runtime.settings.workers_enabled,
            settings=runtime.settings,
        )
        for durable_job in durable_jobs:
            await tracking.enqueue_job(
                runtime.storage_backend,
                durable_job.stream_name,
                durable_job.envelope,
            )
    finally:
        await connection.close()


def _stable_proxy_job_envelope(
    envelope: JobEnvelope,
    *,
    pair_id: str,
) -> JobEnvelope:
    material = "\0".join(
        (
            pair_id,
            envelope.job_type.value,
            *sorted(str(message_id) for message_id in envelope.message_ids),
        )
    ).encode("utf-8")
    job_id = f"job_proxy_{hashlib.sha256(material).hexdigest()[:40]}"
    return envelope.model_copy(update={"job_id": job_id})


def _durable_job_insert(
    runtime: Any,
    *,
    stream_name: str,
    envelope: JobEnvelope,
) -> ProxyDurableJobInsert:
    token_estimate = JobTrackingService._source_token_estimate(envelope)
    metadata = JobTrackingService._safe_metadata(envelope)
    metadata["workers_enabled_at_enqueue"] = runtime.settings.workers_enabled
    snapshot = JobTrackingService._policy_snapshot(envelope)
    return ProxyDurableJobInsert(
        stream_name=stream_name,
        target_backend=runtime.settings.storage_backend,
        envelope=envelope,
        source_token_estimate=token_estimate,
        size_bucket=JobTrackingService._size_bucket(token_estimate),
        metadata=metadata,
        user_persona_id=snapshot.get("user_persona_id"),
        platform_id=snapshot.get("platform_id"),
        character_id=snapshot.get("character_id"),
        incognito_snapshot=bool(snapshot.get("incognito")),
        remember_across_chats_snapshot=bool(
            snapshot.get("remember_across_chats", True)
        ),
        remember_across_devices_snapshot=bool(
            snapshot.get("remember_across_devices", True)
        ),
        temporary_snapshot=bool(snapshot.get("temporary")),
        purge_on_close_snapshot=bool(snapshot.get("purge_on_close")),
        policy_snapshot=snapshot,
    )
