"""Copyable Hermes-style Atagia memory provider adapter.

Hermes-like stacks vary in their provider API. This class exposes small
retrieve/record methods that can be wrapped in the host's actual provider shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from atagia.integrations import (
    MINIMAL_MEMORY_INSTRUCTION,
    SidecarBridge,
    extract_context_system_prompt,
    extract_prompt_data_sections,
    minimal_memory_payload,
)


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
        message_id: str | None = None,
        source_seq: int | None = None,
        memory_privacy_mode: str | None = None,
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
            message_id=message_id,
            source_seq=source_seq,
            ingest_origin="live_turn",
            confirmation_strategy="live_prompt_allowed",
            memory_privacy_mode=memory_privacy_mode,
        )
        payload = minimal_memory_payload(extract_context_system_prompt(context))
        # The host injects this string verbatim as its own system prompt, so
        # the instruction must travel with the data sections rather than being
        # added by ``append_context_to_prompt``. A foreign pass-through payload
        # carries no Atagia data sections and is forwarded unframed, as before.
        if extract_prompt_data_sections(payload):
            payload = f"{MINIMAL_MEMORY_INSTRUCTION}\n\n{payload}"
        return HermesMemoryContext(
            system_prompt=payload,
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
        message_id: str | None = None,
        source_seq: int | None = None,
        memory_privacy_mode: str | None = None,
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
            message_id=message_id,
            source_seq=source_seq,
            ingest_origin="live_turn",
            confirmation_strategy="live_prompt_allowed",
            memory_privacy_mode=memory_privacy_mode,
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
        message_id: str | None = None,
        source_seq: int | None = None,
        memory_privacy_mode: str | None = None,
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
            message_id=message_id,
            source_seq=source_seq,
            ingest_origin="backfill",
            confirmation_strategy="admin_review_only",
            memory_privacy_mode=memory_privacy_mode,
        )

    async def close(self) -> None:
        await self.bridge.close()
