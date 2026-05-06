"""Copyable Hermes-style Atagia memory provider adapter.

Hermes-like stacks vary in their provider API. This class exposes small
retrieve/record methods that can be wrapped in the host's actual provider shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from atagia.integrations import SidecarBridge, extract_context_system_prompt


@dataclass(slots=True)
class HermesMemoryContext:
    """Memory context returned to a Hermes-style host."""

    system_prompt: str
    raw_context: Any | None


class AtagiaHermesProvider:
    """Fail-open memory provider facade over Atagia's sidecar bridge."""

    def __init__(self, bridge: SidecarBridge | None = None) -> None:
        self.bridge = bridge or SidecarBridge()

    async def retrieve(
        self,
        *,
        user_id: str,
        conversation_id: str,
        platform_id: str,
        message: str,
        mode: str = "general_qa",
        user_persona_id: str | None = None,
        character_id: str | None = None,
        incognito: bool | None = None,
    ) -> HermesMemoryContext:
        context = await self.bridge.get_context_for_turn(
            user_id=user_id,
            conversation_id=conversation_id,
            message_text=message,
            mode=mode,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
        )
        return HermesMemoryContext(
            system_prompt=extract_context_system_prompt(context),
            raw_context=context,
        )

    async def record(
        self,
        *,
        user_id: str,
        conversation_id: str,
        platform_id: str,
        assistant_response: str,
        mode: str = "general_qa",
        user_persona_id: str | None = None,
        character_id: str | None = None,
        incognito: bool | None = None,
    ) -> bool:
        return await self.bridge.record_assistant_response(
            user_id=user_id,
            conversation_id=conversation_id,
            response_text=assistant_response,
            mode=mode,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
        )

    async def ingest_historical_message(
        self,
        *,
        user_id: str,
        conversation_id: str,
        platform_id: str,
        role: str,
        text: str,
        mode: str = "general_qa",
        user_persona_id: str | None = None,
        character_id: str | None = None,
        incognito: bool | None = None,
    ) -> bool:
        if role not in {"user", "assistant"}:
            raise ValueError("role must be 'user' or 'assistant'")
        return await self.bridge.ingest_message(
            user_id=user_id,
            conversation_id=conversation_id,
            role=role,
            text=text,
            mode=mode,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=character_id,
            incognito=incognito,
        )

    async def close(self) -> None:
        await self.bridge.close()
