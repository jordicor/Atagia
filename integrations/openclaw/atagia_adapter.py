"""Copyable OpenClaw-style Atagia sidecar adapter.

This file avoids importing OpenClaw internals. Wire the two hook methods into
the host's pre-model and post-model extension points once those hooks are known
for the target install.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from atagia.integrations import SidecarBridge, build_injection_decision


@dataclass(slots=True)
class OpenClawPromptResult:
    """Prompt result returned from an OpenClaw pre-model hook."""

    system_prompt: str
    atagia_active: bool
    atagia_reason: str


class AtagiaOpenClawAdapter:
    """Thin fail-open memory adapter for OpenClaw-style agent hosts."""

    def __init__(self, bridge: SidecarBridge | None = None) -> None:
        self.bridge = bridge or SidecarBridge()

    async def before_model_call(
        self,
        *,
        user_id: str,
        session_id: str,
        agent_id: str,
        platform_id: str,
        user_message: str,
        system_prompt: str,
        attachments: list[dict[str, Any]] | None = None,
        mode: str = "general_qa",
        user_persona_id: str | None = None,
        incognito: bool | None = None,
    ) -> OpenClawPromptResult:
        context = await self.bridge.get_context_for_turn(
            user_id=user_id,
            conversation_id=f"{agent_id}:{session_id}",
            message_text=user_message,
            attachments=attachments,
            mode=mode,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=agent_id,
            incognito=incognito,
        )
        decision = build_injection_decision(system_prompt, context)
        return OpenClawPromptResult(
            system_prompt=decision.full_prompt,
            atagia_active=decision.active,
            atagia_reason=decision.reason,
        )

    async def after_model_call(
        self,
        *,
        user_id: str,
        session_id: str,
        agent_id: str,
        platform_id: str,
        assistant_response: str,
        mode: str = "general_qa",
        user_persona_id: str | None = None,
        incognito: bool | None = None,
    ) -> bool:
        return await self.bridge.record_assistant_response(
            user_id=user_id,
            conversation_id=f"{agent_id}:{session_id}",
            response_text=assistant_response,
            mode=mode,
            user_persona_id=user_persona_id,
            platform_id=platform_id,
            character_id=agent_id,
            incognito=incognito,
        )

    async def close(self) -> None:
        await self.bridge.close()
