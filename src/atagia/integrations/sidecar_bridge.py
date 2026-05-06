"""Generic fail-open bridge for host-managed Atagia sidecar integrations."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
import logging
import os
import uuid
from typing import Any, Literal, Protocol

from atagia.models.schemas_jobs import WorkerControlMode
from atagia.models.schemas_memory import (
    ConfirmationStrategy,
    IngestOrigin,
    MemoryPrivacyMode,
)

logger = logging.getLogger(__name__)

TransportName = Literal["auto", "local", "http"]
DEFAULT_MODE = "personal_assistant"
DEFAULT_TIMEOUT_SECONDS = 30.0
_TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}
_FALSE_VALUES = {"0", "false", "no", "off", "disabled"}
_VALID_TRANSPORTS: set[str] = {"auto", "local", "http"}


class AtagiaClientProtocol(Protocol):
    """Subset of the generic Atagia client used by host adapters."""

    async def create_user(self, user_id: str) -> None:
        """Create the user if needed."""

    async def create_conversation(
        self,
        user_id: str,
        conversation_id: str | None,
        *,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        mode: str | None = None,
        incognito: bool | None = None,
    ) -> str:
        """Create or reuse an Atagia conversation."""

    async def get_context(
        self,
        user_id: str,
        conversation_id: str,
        message: str,
        mode: str | None = None,
        occurred_at: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        message_id: str | None = None,
        source_seq: int | str | None = None,
        *,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool | None = None,
        ingest_origin: str | None = None,
        confirmation_strategy: str | None = None,
        memory_privacy_mode: str | None = None,
    ) -> Any:
        """Return memory context for a host-managed LLM call."""

    async def add_response(
        self,
        user_id: str,
        conversation_id: str,
        text: str,
        occurred_at: str | None = None,
        *,
        message_id: str | None = None,
        source_seq: int | str | None = None,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        mode: str | None = None,
        incognito: bool | None = None,
        ingest_origin: str | None = None,
        confirmation_strategy: str | None = None,
        memory_privacy_mode: str | None = None,
    ) -> None:
        """Persist a host-generated assistant response."""

    async def ingest_message(
        self,
        user_id: str,
        conversation_id: str,
        role: Literal["user", "assistant"],
        text: str,
        mode: str | None = None,
        occurred_at: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        message_id: str | None = None,
        source_seq: int | str | None = None,
        *,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        incognito: bool | None = None,
        ingest_origin: str | None = None,
        confirmation_strategy: str | None = None,
        memory_privacy_mode: str | None = None,
    ) -> None:
        """Persist one historical or sidecar message."""

    async def close(self) -> None:
        """Close transport resources."""

    async def flush(
        self,
        timeout_seconds: float = 30.0,
        user_id: str | None = None,
    ) -> bool:
        """Wait for pending sidecar work to finish."""

    async def list_pending_memory_confirmations(
        self,
        user_id: str,
        **filters: Any,
    ) -> Any:
        """Return pending user memory confirmations."""

    async def confirm_pending_memory(self, user_id: str, memory_id: str) -> Any:
        """Confirm one pending user memory."""

    async def decline_pending_memory(self, user_id: str, memory_id: str) -> Any:
        """Decline one pending user memory."""

    async def list_review_required_memories(self, **filters: Any) -> Any:
        """Return admin review-required memories."""

    async def archive_review_required_memory(self, user_id: str, memory_id: str) -> Any:
        """Archive one review-required memory."""

    async def delete_review_required_memory(self, user_id: str, memory_id: str) -> Any:
        """Delete one review-required memory."""

    async def get_worker_control(self) -> Any:
        """Return current background-processing control state."""

    async def set_worker_control(
        self,
        mode: WorkerControlMode | str,
        *,
        reason: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> Any:
        """Set the background-processing control state."""


ClientFactory = Callable[..., Awaitable[AtagiaClientProtocol]]
ConfigLoader = Callable[[], Awaitable["SidecarBridgeConfig"]]


@dataclass(frozen=True, slots=True)
class SidecarBridgeConfig:
    """Settings for a host application's Atagia sidecar bridge."""

    enabled: bool = False
    transport: TransportName = "auto"
    db_path: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    admin_api_key: str | None = None
    mode: str = DEFAULT_MODE
    user_persona_id: str | None = None
    platform_id: str | None = None
    character_id: str | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    operational_profile: str | None = None
    operational_signals: dict[str, Any] = field(default_factory=dict)
    incognito: bool = False
    memory_privacy_mode: str | None = None

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> "SidecarBridgeConfig":
        env = environ or os.environ
        transport = _parse_transport(env.get("ATAGIA_TRANSPORT", "auto"))
        return cls(
            enabled=_parse_bool(env.get("ATAGIA_ENABLED")),
            transport=transport,
            db_path=_clean_optional(
                env.get("ATAGIA_DB_PATH") or env.get("ATAGIA_SQLITE_PATH")
            ),
            base_url=_clean_optional(env.get("ATAGIA_BASE_URL")),
            api_key=_clean_optional(env.get("ATAGIA_SERVICE_API_KEY")),
            admin_api_key=_clean_optional(env.get("ATAGIA_ADMIN_API_KEY")),
            mode=(_clean_optional(env.get("ATAGIA_MODE")) or DEFAULT_MODE),
            user_persona_id=_clean_optional(env.get("ATAGIA_USER_PERSONA_ID")),
            platform_id=_clean_optional(env.get("ATAGIA_PLATFORM_ID")),
            character_id=_clean_optional(env.get("ATAGIA_CHARACTER_ID")),
            timeout_seconds=_parse_timeout(env.get("ATAGIA_TIMEOUT_SECONDS")),
            operational_profile=_clean_optional(
                env.get("ATAGIA_OPERATIONAL_PROFILE")
            ),
            incognito=_parse_bool(env.get("ATAGIA_INCOGNITO")),
            memory_privacy_mode=_clean_optional(
                env.get("ATAGIA_MEMORY_PRIVACY_MODE")
            ),
        )


@dataclass(frozen=True, slots=True)
class SidecarBridgeError:
    """Structured details for the most recent fail-open sidecar error."""

    operation: str
    error_type: str
    message: str
    status_code: int | None = None
    details: Any | None = None


class SidecarBridge:
    """Host-application Atagia adapter with fail-open behavior."""

    def __init__(
        self,
        config: SidecarBridgeConfig | None = None,
        *,
        client_factory: ClientFactory | None = None,
        config_loader: ConfigLoader | None = None,
    ) -> None:
        self.config = config or SidecarBridgeConfig.from_env()
        self._client_factory = client_factory or _default_client_factory
        self._config_loader = config_loader
        self._client: AtagiaClientProtocol | None = None
        self._last_error: SidecarBridgeError | None = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def last_error(self) -> SidecarBridgeError | None:
        """Structured details for the most recent fail-open operation error."""
        return self._last_error

    async def ensure_user_and_conversation(
        self,
        user_id: int | str,
        conversation_id: int | str,
        *,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        mode: str | None = None,
        incognito: bool | None = None,
    ) -> str | None:
        """Ensure Atagia resources exist, returning the conversation id."""
        config = await self._get_config()
        if not config.enabled:
            self._last_error = None
            return None
        try:
            client = await self._ensure_client(config)
            atagia_user_id = _to_atagia_id(user_id)
            atagia_conversation_id = _to_atagia_id(conversation_id)
            resolved_platform_id = _resolve_platform_id(platform_id, config)
            await client.create_user(atagia_user_id)
            conversation_id_result = await client.create_conversation(
                user_id=atagia_user_id,
                conversation_id=atagia_conversation_id,
                user_persona_id=user_persona_id or config.user_persona_id,
                platform_id=resolved_platform_id,
                character_id=character_id or config.character_id,
                mode=mode or config.mode,
                incognito=_resolve_incognito(incognito, config),
            )
            self._last_error = None
            return conversation_id_result
        except Exception as exc:
            self._last_error = _bridge_error("ensure_user_and_conversation", exc)
            logger.warning(
                "Atagia ensure_user_and_conversation failed; "
                "continuing without sidecar memory",
                exc_info=True,
            )
            return None

    async def get_context_for_turn(
        self,
        user_id: int | str,
        conversation_id: int | str,
        message_text: str,
        *,
        occurred_at: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        mode: str | None = None,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
        incognito: bool | None = None,
        message_id: int | str | None = None,
        source_seq: int | str | None = None,
        ingest_origin: IngestOrigin | str | None = None,
        confirmation_strategy: ConfirmationStrategy | str | None = None,
        memory_privacy_mode: MemoryPrivacyMode | str | None = None,
    ) -> Any | None:
        """Return Atagia context for one user turn, or None on disabled/error."""
        config = await self._get_config()
        if not config.enabled or not message_text:
            self._last_error = None
            return None
        try:
            client = await self._ensure_client(config)
            resolved_platform_id = _resolve_platform_id(platform_id, config)
            context_result = await client.get_context(
                user_id=_to_atagia_id(user_id),
                conversation_id=_to_atagia_id(conversation_id),
                message=message_text,
                mode=mode or config.mode,
                occurred_at=occurred_at,
                attachments=attachments,
                message_id=_to_optional_atagia_id(message_id),
                source_seq=_to_optional_source_seq(source_seq),
                operational_profile=operational_profile
                or config.operational_profile,
                operational_signals=operational_signals
                if operational_signals is not None
                else config.operational_signals,
                user_persona_id=user_persona_id or config.user_persona_id,
                platform_id=resolved_platform_id,
                character_id=character_id or config.character_id,
                incognito=_resolve_incognito(incognito, config),
                ingest_origin=_to_optional_ingest_origin(ingest_origin),
                confirmation_strategy=_to_optional_confirmation_strategy(
                    confirmation_strategy
                ),
                memory_privacy_mode=_to_optional_memory_privacy_mode(
                    memory_privacy_mode or config.memory_privacy_mode
                ),
            )
            self._last_error = None
            return context_result
        except Exception as exc:
            self._last_error = _bridge_error("get_context_for_turn", exc)
            logger.warning(
                "Atagia get_context failed; falling back to host context",
                exc_info=True,
            )
            return None

    async def ingest_message(
        self,
        user_id: int | str,
        conversation_id: int | str,
        role: Literal["user", "assistant"],
        text: str,
        *,
        occurred_at: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        mode: str | None = None,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
        incognito: bool | None = None,
        message_id: int | str | None = None,
        source_seq: int | str | None = None,
        ingest_origin: IngestOrigin | str | None = None,
        confirmation_strategy: ConfirmationStrategy | str | None = None,
        memory_privacy_mode: MemoryPrivacyMode | str | None = None,
    ) -> bool:
        """Persist a host-managed message in Atagia, returning success."""
        config = await self._get_config()
        if not config.enabled or not text:
            self._last_error = None
            return False
        try:
            client = await self._ensure_client(config)
            resolved_platform_id = _resolve_platform_id(platform_id, config)
            await client.ingest_message(
                user_id=_to_atagia_id(user_id),
                conversation_id=_to_atagia_id(conversation_id),
                role=role,
                text=text,
                mode=mode or config.mode,
                occurred_at=occurred_at,
                attachments=attachments,
                message_id=_to_optional_atagia_id(message_id),
                source_seq=_to_optional_source_seq(source_seq),
                operational_profile=operational_profile
                or config.operational_profile,
                operational_signals=operational_signals
                if operational_signals is not None
                else config.operational_signals,
                user_persona_id=user_persona_id or config.user_persona_id,
                platform_id=resolved_platform_id,
                character_id=character_id or config.character_id,
                incognito=_resolve_incognito(incognito, config),
                ingest_origin=_to_optional_ingest_origin(ingest_origin),
                confirmation_strategy=_to_optional_confirmation_strategy(
                    confirmation_strategy
                ),
                memory_privacy_mode=_to_optional_memory_privacy_mode(
                    memory_privacy_mode or config.memory_privacy_mode
                ),
            )
            self._last_error = None
            return True
        except Exception as exc:
            self._last_error = _bridge_error("ingest_message", exc)
            logger.warning(
                "Atagia ingest_message failed; continuing without sidecar persistence",
                exc_info=True,
            )
            return False

    async def record_assistant_response(
        self,
        user_id: int | str,
        conversation_id: int | str,
        response_text: str,
        *,
        occurred_at: str | None = None,
        user_persona_id: str | None = None,
        platform_id: str | None = None,
        character_id: str | None = None,
        mode: str | None = None,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
        incognito: bool | None = None,
        message_id: int | str | None = None,
        source_seq: int | str | None = None,
        ingest_origin: IngestOrigin | str | None = None,
        confirmation_strategy: ConfirmationStrategy | str | None = None,
        memory_privacy_mode: MemoryPrivacyMode | str | None = None,
    ) -> bool:
        """Persist the assistant response in Atagia, returning success."""
        config = await self._get_config()
        if not config.enabled or not response_text:
            self._last_error = None
            return False
        try:
            client = await self._ensure_client(config)
            resolved_platform_id = _resolve_platform_id(platform_id, config)
            await client.add_response(
                user_id=_to_atagia_id(user_id),
                conversation_id=_to_atagia_id(conversation_id),
                text=response_text,
                occurred_at=occurred_at,
                message_id=_to_optional_atagia_id(message_id),
                source_seq=_to_optional_source_seq(source_seq),
                operational_profile=operational_profile
                or config.operational_profile,
                operational_signals=operational_signals
                if operational_signals is not None
                else config.operational_signals,
                user_persona_id=user_persona_id or config.user_persona_id,
                platform_id=resolved_platform_id,
                character_id=character_id or config.character_id,
                mode=mode or config.mode,
                incognito=_resolve_incognito(incognito, config),
                ingest_origin=_to_optional_ingest_origin(ingest_origin),
                confirmation_strategy=_to_optional_confirmation_strategy(
                    confirmation_strategy
                ),
                memory_privacy_mode=_to_optional_memory_privacy_mode(
                    memory_privacy_mode or config.memory_privacy_mode
                ),
            )
            self._last_error = None
            return True
        except Exception as exc:
            self._last_error = _bridge_error("record_assistant_response", exc)
            logger.warning(
                "Atagia add_response failed; continuing without sidecar persistence",
                exc_info=True,
            )
            return False

    async def test_connection(self) -> tuple[bool, str]:
        """Best-effort admin connection check using real Atagia resources."""
        config = await self._get_config()
        if not config.enabled:
            self._last_error = None
            return False, "Atagia is disabled."
        try:
            client = await self._ensure_client(config)
            test_user_id = "atagia-sidecar-test"
            test_conversation_id = f"atagia-sidecar-test-{uuid.uuid4().hex}"
            await client.create_user(test_user_id)
            await client.create_conversation(
                user_id=test_user_id,
                conversation_id=test_conversation_id,
                user_persona_id=config.user_persona_id,
                platform_id=_resolve_platform_id(None, config),
                character_id=config.character_id,
                mode=config.mode,
                incognito=config.incognito,
            )
            self._last_error = None
            return True, "Atagia connection successful."
        except Exception as exc:
            self._last_error = _bridge_error("test_connection", exc)
            logger.warning("Atagia connection test failed", exc_info=True)
            return False, str(exc)

    async def flush(
        self,
        timeout_seconds: float | None = None,
        *,
        user_id: int | str | None = None,
    ) -> bool:
        """Wait for pending Atagia sidecar work, returning success."""
        config = await self._get_config()
        if not config.enabled:
            self._last_error = None
            return False
        try:
            client = await self._ensure_client(config)
            completed = await client.flush(
                timeout_seconds or config.timeout_seconds,
                user_id=_to_optional_atagia_id(user_id),
            )
            self._last_error = None
            return completed
        except Exception as exc:
            self._last_error = _bridge_error("flush", exc)
            logger.warning(
                "Atagia flush failed; continuing without blocking host app",
                exc_info=True,
            )
            return False

    async def list_pending_memory_confirmations(
        self,
        user_id: int | str,
        **filters: Any,
    ) -> Any | None:
        """Return pending user confirmations, or None on disabled/error."""
        config = await self._get_config()
        if not config.enabled:
            self._last_error = None
            return None
        try:
            client = await self._ensure_client(config)
            result = await client.list_pending_memory_confirmations(
                _to_atagia_id(user_id),
                **filters,
            )
            self._last_error = None
            return result
        except Exception as exc:
            self._last_error = _bridge_error("list_pending_memory_confirmations", exc)
            logger.warning(
                "Atagia list_pending_memory_confirmations failed; continuing host operation",
                exc_info=True,
            )
            return None

    async def confirm_pending_memory(
        self,
        user_id: int | str,
        memory_id: int | str,
    ) -> Any | None:
        """Confirm one pending memory, or None on disabled/error."""
        config = await self._get_config()
        if not config.enabled:
            self._last_error = None
            return None
        try:
            client = await self._ensure_client(config)
            result = await client.confirm_pending_memory(
                _to_atagia_id(user_id),
                _to_atagia_id(memory_id),
            )
            self._last_error = None
            return result
        except Exception as exc:
            self._last_error = _bridge_error("confirm_pending_memory", exc)
            logger.warning(
                "Atagia confirm_pending_memory failed; continuing host operation",
                exc_info=True,
            )
            return None

    async def decline_pending_memory(
        self,
        user_id: int | str,
        memory_id: int | str,
    ) -> Any | None:
        """Decline one pending memory, or None on disabled/error."""
        config = await self._get_config()
        if not config.enabled:
            self._last_error = None
            return None
        try:
            client = await self._ensure_client(config)
            result = await client.decline_pending_memory(
                _to_atagia_id(user_id),
                _to_atagia_id(memory_id),
            )
            self._last_error = None
            return result
        except Exception as exc:
            self._last_error = _bridge_error("decline_pending_memory", exc)
            logger.warning(
                "Atagia decline_pending_memory failed; continuing host operation",
                exc_info=True,
            )
            return None

    async def list_review_required_memories(self, **filters: Any) -> Any | None:
        """Return admin review-required memories, or None on disabled/error."""
        config = await self._get_config()
        if not config.enabled:
            self._last_error = None
            return None
        try:
            client = await self._ensure_client(config)
            result = await client.list_review_required_memories(**filters)
            self._last_error = None
            return result
        except Exception as exc:
            self._last_error = _bridge_error("list_review_required_memories", exc)
            logger.warning(
                "Atagia list_review_required_memories failed; continuing host operation",
                exc_info=True,
            )
            return None

    async def archive_review_required_memory(
        self,
        user_id: int | str,
        memory_id: int | str,
    ) -> Any | None:
        """Archive one review-required memory, or None on disabled/error."""
        config = await self._get_config()
        if not config.enabled:
            self._last_error = None
            return None
        try:
            client = await self._ensure_client(config)
            result = await client.archive_review_required_memory(
                _to_atagia_id(user_id),
                _to_atagia_id(memory_id),
            )
            self._last_error = None
            return result
        except Exception as exc:
            self._last_error = _bridge_error("archive_review_required_memory", exc)
            logger.warning(
                "Atagia archive_review_required_memory failed; continuing host operation",
                exc_info=True,
            )
            return None

    async def delete_review_required_memory(
        self,
        user_id: int | str,
        memory_id: int | str,
    ) -> Any | None:
        """Delete one review-required memory, or None on disabled/error."""
        config = await self._get_config()
        if not config.enabled:
            self._last_error = None
            return None
        try:
            client = await self._ensure_client(config)
            result = await client.delete_review_required_memory(
                _to_atagia_id(user_id),
                _to_atagia_id(memory_id),
            )
            self._last_error = None
            return result
        except Exception as exc:
            self._last_error = _bridge_error("delete_review_required_memory", exc)
            logger.warning(
                "Atagia delete_review_required_memory failed; continuing host operation",
                exc_info=True,
            )
            return None

    async def get_worker_control(self) -> Any | None:
        """Return the Atagia processing stop-switch state, or None on error."""
        config = await self._get_config()
        if not config.enabled:
            self._last_error = None
            return None
        try:
            client = await self._ensure_client(config)
            state = await client.get_worker_control()
            self._last_error = None
            return state
        except Exception as exc:
            self._last_error = _bridge_error("get_worker_control", exc)
            logger.warning(
                "Atagia get_worker_control failed; continuing host operation",
                exc_info=True,
            )
            return None

    async def set_worker_control(
        self,
        mode: WorkerControlMode | str,
        *,
        reason: str | None = None,
        timeout_seconds: float | None = None,
    ) -> Any | None:
        """Set the Atagia processing stop-switch state, fail-open on error."""
        config = await self._get_config()
        if not config.enabled:
            self._last_error = None
            return None
        try:
            client = await self._ensure_client(config)
            state = await client.set_worker_control(
                WorkerControlMode(mode),
                reason=reason,
                timeout_seconds=timeout_seconds or config.timeout_seconds,
            )
            self._last_error = None
            return state
        except Exception as exc:
            self._last_error = _bridge_error("set_worker_control", exc)
            logger.warning(
                "Atagia set_worker_control failed; continuing host operation",
                exc_info=True,
            )
            return None

    async def pause_new_jobs(self, *, reason: str | None = None) -> Any | None:
        """Store host messages without adding new memory-processing jobs."""
        return await self.set_worker_control(
            WorkerControlMode.PAUSE_NEW_JOBS,
            reason=reason,
        )

    async def drain_and_pause(
        self,
        *,
        reason: str | None = None,
        timeout_seconds: float | None = None,
    ) -> Any | None:
        """Stop new source jobs, wait for queued work, and remain paused."""
        return await self.set_worker_control(
            WorkerControlMode.DRAIN_AND_PAUSE,
            reason=reason,
            timeout_seconds=timeout_seconds,
        )

    async def hard_pause(self, *, reason: str | None = None) -> Any | None:
        """Stop workers from claiming any further stream jobs."""
        return await self.set_worker_control(
            WorkerControlMode.HARD_PAUSE,
            reason=reason,
        )

    async def resume_processing(self, *, reason: str | None = None) -> Any | None:
        """Resume normal Atagia background processing."""
        return await self.set_worker_control(
            WorkerControlMode.ACTIVE,
            reason=reason,
        )

    async def close(self) -> None:
        """Close the initialized Atagia client, if any."""
        if self._client is None:
            return
        await self._client.close()
        self._client = None

    async def _get_config(self) -> SidecarBridgeConfig:
        if self._config_loader is None:
            return self.config
        loaded = await self._config_loader()
        if loaded != self.config:
            await self.close()
            self.config = loaded
        return self.config

    async def _ensure_client(
        self,
        config: SidecarBridgeConfig,
    ) -> AtagiaClientProtocol:
        if self._client is None:
            kwargs: dict[str, Any] = {
                "transport": config.transport,
                "db_path": config.db_path,
                "base_url": config.base_url,
                "api_key": config.api_key,
                "timeout": config.timeout_seconds,
            }
            if config.admin_api_key is not None:
                kwargs["admin_api_key"] = config.admin_api_key
            self._client = await self._client_factory(**kwargs)
        return self._client


async def _default_client_factory(**kwargs: Any) -> AtagiaClientProtocol:
    from atagia.client import connect_atagia

    return await connect_atagia(**kwargs)


def _parse_bool(value: str | None) -> bool:
    return (value or "").strip().lower() in _TRUE_VALUES


def _parse_bool_default(value: str | None, default: bool) -> bool:
    normalized = (value or "").strip().lower()
    if not normalized:
        return default
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    logger.warning("Invalid boolean value %r; using default %s", value, default)
    return default


def _resolve_platform_id(
    explicit: str | None,
    config: SidecarBridgeConfig,
) -> str:
    platform_id = _clean_optional(explicit) or config.platform_id
    if platform_id is None:
        raise ValueError("platform_id is required for sidecar integrations")
    return platform_id


def _resolve_incognito(
    explicit: bool | None,
    config: SidecarBridgeConfig,
) -> bool:
    return config.incognito if explicit is None else bool(explicit)


def _parse_transport(value: str | None) -> TransportName:
    transport = (value or "auto").strip().lower()
    if transport not in _VALID_TRANSPORTS:
        logger.warning("Unknown ATAGIA_TRANSPORT=%r; using auto", value)
        return "auto"
    return transport  # type: ignore[return-value]


def _parse_timeout(value: str | None) -> float:
    if value is None or not value.strip():
        return DEFAULT_TIMEOUT_SECONDS
    try:
        timeout = float(value)
    except ValueError:
        logger.warning("Invalid ATAGIA_TIMEOUT_SECONDS=%r; using default", value)
        return DEFAULT_TIMEOUT_SECONDS
    if timeout <= 0:
        logger.warning("Invalid ATAGIA_TIMEOUT_SECONDS=%r; using default", value)
        return DEFAULT_TIMEOUT_SECONDS
    return timeout


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _to_atagia_id(value: int | str) -> str:
    return str(value)


def _to_optional_atagia_id(value: int | str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _to_optional_source_seq(value: int | str | None) -> int | None:
    if value is None:
        return None
    resolved = int(value)
    if resolved < 1:
        raise ValueError("source_seq must be a positive integer")
    return resolved


def _to_optional_ingest_origin(value: IngestOrigin | str | None) -> str | None:
    if value is None:
        return None
    return IngestOrigin(value).value


def _to_optional_confirmation_strategy(
    value: ConfirmationStrategy | str | None,
) -> str | None:
    if value is None:
        return None
    return ConfirmationStrategy(value).value


def _to_optional_memory_privacy_mode(
    value: MemoryPrivacyMode | str | None,
) -> str | None:
    if value is None:
        return None
    return MemoryPrivacyMode(value).value


def _bridge_error(operation: str, exc: Exception) -> SidecarBridgeError:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    details: Any | None = None
    if response is not None:
        try:
            payload = response.json()
        except Exception:
            payload = getattr(response, "text", None)
        if isinstance(payload, dict) and "detail" in payload:
            details = payload["detail"]
        else:
            details = payload
    message = str(details or exc)
    return SidecarBridgeError(
        operation=operation,
        error_type=exc.__class__.__name__,
        message=message,
        status_code=status_code if isinstance(status_code, int) else None,
        details=details,
    )
