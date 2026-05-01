"""Generic fail-open bridge for host-managed Atagia sidecar integrations."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
import logging
import os
import uuid
from typing import Any, Literal, Protocol

logger = logging.getLogger(__name__)

TransportName = Literal["auto", "local", "http"]
DEFAULT_ASSISTANT_MODE = "personal_assistant"
DEFAULT_TIMEOUT_SECONDS = 30.0
_TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}
_VALID_TRANSPORTS: set[str] = {"auto", "local", "http"}


class AtagiaClientProtocol(Protocol):
    """Subset of the generic Atagia client used by host adapters."""

    async def create_user(self, user_id: str) -> None:
        """Create the user if needed."""

    async def create_conversation(
        self,
        user_id: str,
        conversation_id: str | None,
        workspace_id: str | None = None,
        assistant_mode_id: str | None = None,
    ) -> str:
        """Create or reuse an Atagia conversation."""

    async def get_context(
        self,
        user_id: str,
        conversation_id: str,
        message: str,
        mode: str | None = None,
        workspace_id: str | None = None,
        occurred_at: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        *,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
    ) -> Any:
        """Return memory context for a host-managed LLM call."""

    async def add_response(
        self,
        user_id: str,
        conversation_id: str,
        text: str,
        occurred_at: str | None = None,
        *,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
    ) -> None:
        """Persist a host-generated assistant response."""

    async def ingest_message(
        self,
        user_id: str,
        conversation_id: str,
        role: Literal["user", "assistant"],
        text: str,
        mode: str | None = None,
        workspace_id: str | None = None,
        occurred_at: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        *,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
    ) -> None:
        """Persist one historical or sidecar message."""

    async def close(self) -> None:
        """Close transport resources."""


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
    assistant_mode: str = DEFAULT_ASSISTANT_MODE
    workspace_id: str | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    operational_profile: str | None = None
    operational_signals: dict[str, Any] = field(default_factory=dict)

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
            assistant_mode=(
                _clean_optional(env.get("ATAGIA_ASSISTANT_MODE"))
                or DEFAULT_ASSISTANT_MODE
            ),
            workspace_id=_clean_optional(env.get("ATAGIA_WORKSPACE_ID")),
            timeout_seconds=_parse_timeout(env.get("ATAGIA_TIMEOUT_SECONDS")),
            operational_profile=_clean_optional(
                env.get("ATAGIA_OPERATIONAL_PROFILE")
            ),
        )


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

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    async def ensure_user_and_conversation(
        self,
        user_id: int | str,
        conversation_id: int | str,
        *,
        workspace_id: str | None = None,
        assistant_mode: str | None = None,
    ) -> str | None:
        """Ensure Atagia resources exist, returning the conversation id."""
        config = await self._get_config()
        if not config.enabled:
            return None
        try:
            client = await self._ensure_client(config)
            atagia_user_id = _to_atagia_id(user_id)
            atagia_conversation_id = _to_atagia_id(conversation_id)
            await client.create_user(atagia_user_id)
            return await client.create_conversation(
                user_id=atagia_user_id,
                conversation_id=atagia_conversation_id,
                workspace_id=workspace_id or config.workspace_id,
                assistant_mode_id=assistant_mode or config.assistant_mode,
            )
        except Exception:
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
        workspace_id: str | None = None,
        assistant_mode: str | None = None,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
    ) -> Any | None:
        """Return Atagia context for one user turn, or None on disabled/error."""
        config = await self._get_config()
        if not config.enabled or not message_text:
            return None
        try:
            client = await self._ensure_client(config)
            return await client.get_context(
                user_id=_to_atagia_id(user_id),
                conversation_id=_to_atagia_id(conversation_id),
                message=message_text,
                mode=assistant_mode or config.assistant_mode,
                workspace_id=workspace_id or config.workspace_id,
                occurred_at=occurred_at,
                attachments=attachments,
                operational_profile=operational_profile
                or config.operational_profile,
                operational_signals=operational_signals
                if operational_signals is not None
                else config.operational_signals,
            )
        except Exception:
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
        workspace_id: str | None = None,
        assistant_mode: str | None = None,
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
    ) -> bool:
        """Persist a host-managed message in Atagia, returning success."""
        config = await self._get_config()
        if not config.enabled or not text:
            return False
        try:
            client = await self._ensure_client(config)
            await client.ingest_message(
                user_id=_to_atagia_id(user_id),
                conversation_id=_to_atagia_id(conversation_id),
                role=role,
                text=text,
                mode=assistant_mode or config.assistant_mode,
                workspace_id=workspace_id or config.workspace_id,
                occurred_at=occurred_at,
                attachments=attachments,
                operational_profile=operational_profile
                or config.operational_profile,
                operational_signals=operational_signals
                if operational_signals is not None
                else config.operational_signals,
            )
            return True
        except Exception:
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
        operational_profile: str | None = None,
        operational_signals: dict[str, Any] | None = None,
    ) -> bool:
        """Persist the assistant response in Atagia, returning success."""
        config = await self._get_config()
        if not config.enabled or not response_text:
            return False
        try:
            client = await self._ensure_client(config)
            await client.add_response(
                user_id=_to_atagia_id(user_id),
                conversation_id=_to_atagia_id(conversation_id),
                text=response_text,
                occurred_at=occurred_at,
                operational_profile=operational_profile
                or config.operational_profile,
                operational_signals=operational_signals
                if operational_signals is not None
                else config.operational_signals,
            )
            return True
        except Exception:
            logger.warning(
                "Atagia add_response failed; continuing without sidecar persistence",
                exc_info=True,
            )
            return False

    async def test_connection(self) -> tuple[bool, str]:
        """Best-effort admin connection check using real Atagia resources."""
        config = await self._get_config()
        if not config.enabled:
            return False, "Atagia is disabled."
        try:
            client = await self._ensure_client(config)
            test_user_id = "atagia-sidecar-test"
            test_conversation_id = f"atagia-sidecar-test-{uuid.uuid4().hex}"
            await client.create_user(test_user_id)
            await client.create_conversation(
                user_id=test_user_id,
                conversation_id=test_conversation_id,
                workspace_id=config.workspace_id,
                assistant_mode_id=config.assistant_mode,
            )
            return True, "Atagia connection successful."
        except Exception as exc:
            logger.warning("Atagia connection test failed", exc_info=True)
            return False, str(exc)

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
            self._client = await self._client_factory(
                transport=config.transport,
                db_path=config.db_path,
                base_url=config.base_url,
                api_key=config.api_key,
                timeout=config.timeout_seconds,
            )
        return self._client


async def _default_client_factory(**kwargs: Any) -> AtagiaClientProtocol:
    from atagia.client import connect_atagia

    return await connect_atagia(**kwargs)


def _parse_bool(value: str | None) -> bool:
    return (value or "").strip().lower() in _TRUE_VALUES


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
