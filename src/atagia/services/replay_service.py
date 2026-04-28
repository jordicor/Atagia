"""Replay helpers for retrieval-only analysis."""

from __future__ import annotations

from typing import Any

import aiosqlite

from atagia.core.clock import Clock
from atagia.core.config import Settings
from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, MessageRepository
from atagia.core.retrieval_event_repository import RetrievalEventRepository
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver
from atagia.memory.retrieval_comparator import RetrievalComparator
from atagia.models.schemas_memory import ExtractionContextMessage, ExtractionConversationContext
from atagia.models.schemas_replay import AblationConfig, ReplayResult
from atagia.services.retrieval_pipeline import RetrievalPipeline

RECENT_CONTEXT_MESSAGES = 6


class ReplayService:
    """Replay retrieval traces against current memory state.

    Known limitation: replays use current memory state, not the state at the time
    of the original retrieval event.
    """

    def __init__(
        self,
        connection: aiosqlite.Connection,
        retrieval_pipeline: RetrievalPipeline,
        clock: Clock,
        settings: Settings | None = None,
    ) -> None:
        self._connection = connection
        self._retrieval_pipeline = retrieval_pipeline
        self._clock = clock
        self._settings = settings or Settings.from_env()
        self._conversation_repository = ConversationRepository(connection, clock)
        self._message_repository = MessageRepository(connection, clock)
        self._memory_repository = MemoryObjectRepository(connection, clock)
        self._retrieval_event_repository = RetrievalEventRepository(connection, clock)
        self._manifest_loader = ManifestLoader(self._settings.manifests_dir())
        self._policy_resolver = PolicyResolver()
        self._comparator = RetrievalComparator()

    async def replay_retrieval_event(
        self,
        event_id: str,
        user_id: str,
        ablation: AblationConfig | None = None,
    ) -> ReplayResult:
        event = await self._retrieval_event_repository.get_event(event_id, user_id)
        if event is None:
            raise ValueError("Retrieval event not found for user")

        conversation = await self._conversation_repository.get_conversation(
            str(event["conversation_id"]),
            user_id,
        )
        if conversation is None:
            raise ValueError("Conversation not found for user")

        transcript, conversation_context, message_text = await self._reconstruct_context(
            conversation_id=str(conversation["id"]),
            user_id=user_id,
            request_message_id=str(event["request_message_id"]),
            workspace_id=conversation.get("workspace_id"),
            assistant_mode_id=str(conversation["assistant_mode_id"]),
        )
        resolved_policy = self._resolve_policy(str(conversation["assistant_mode_id"]))
        cold_start = await self._is_cold_start(
            user_id=user_id,
            conversation_id=str(conversation["id"]),
            workspace_id=conversation.get("workspace_id"),
            assistant_mode_id=str(conversation["assistant_mode_id"]),
            resolved_policy=resolved_policy,
        )
        replay_pipeline_result = await self._retrieval_pipeline.execute(
            message_text=message_text,
            conversation_context=conversation_context,
            resolved_policy=resolved_policy,
            cold_start=cold_start,
            ablation=ablation,
            conversation_messages=transcript,
        )
        comparison = self._comparator.compare(event, replay_pipeline_result)
        return ReplayResult(
            original_event_id=event_id,
            replay_pipeline_result=replay_pipeline_result,
            comparison=comparison,
            ablation_config=(
                ablation.model_dump(mode="json", exclude_none=True)
                if ablation is not None
                else None
            ),
        )

    async def replay_conversation(
        self,
        conversation_id: str,
        user_id: str,
        ablation: AblationConfig | None = None,
        message_limit: int | None = None,
    ) -> list[ReplayResult]:
        conversation = await self._conversation_repository.get_conversation(conversation_id, user_id)
        if conversation is None:
            raise ValueError("Conversation not found for user")

        messages = await self._message_repository.get_messages(conversation_id, user_id, limit=5000, offset=0)
        events = await self._retrieval_event_repository.list_events(user_id, conversation_id, limit=5000, offset=0)
        event_by_request_message_id = {
            str(event["request_message_id"]): str(event["id"])
            for event in events
        }
        replay_results: list[ReplayResult] = []
        for message in messages:
            if str(message["role"]) != "user":
                continue
            event_id = event_by_request_message_id.get(str(message["id"]))
            if event_id is None:
                continue
            replay_results.append(
                await self.replay_retrieval_event(
                    event_id=event_id,
                    user_id=user_id,
                    ablation=ablation,
                )
            )
            if message_limit is not None and len(replay_results) >= message_limit:
                break
        return replay_results

    async def _reconstruct_context(
        self,
        *,
        conversation_id: str,
        user_id: str,
        request_message_id: str,
        workspace_id: str | None,
        assistant_mode_id: str,
    ) -> tuple[list[dict[str, Any]], ExtractionConversationContext, str]:
        messages = await self._message_repository.get_messages(conversation_id, user_id, limit=5000, offset=0)
        request_message = next((message for message in messages if str(message["id"]) == request_message_id), None)
        if request_message is None:
            raise ValueError("Request message not found for user")
        request_seq = int(request_message["seq"])
        transcript = [message for message in messages if int(message["seq"]) <= request_seq]
        prior_messages = [message for message in transcript if int(message["seq"]) < request_seq]
        conversation_context = ExtractionConversationContext(
            user_id=user_id,
            conversation_id=conversation_id,
            source_message_id=request_message_id,
            workspace_id=workspace_id,
            assistant_mode_id=assistant_mode_id,
            recent_messages=[
                ExtractionContextMessage(
                    role=str(message["role"]),
                    content=str(message["text"]),
                )
                for message in prior_messages[-RECENT_CONTEXT_MESSAGES:]
            ],
        )
        return transcript, conversation_context, str(request_message["text"])

    def _resolve_policy(self, assistant_mode_id: str):
        manifest = self._manifest_loader.get(assistant_mode_id)
        return self._policy_resolver.resolve(manifest, None, None)

    async def _is_cold_start(
        self,
        *,
        user_id: str,
        conversation_id: str,
        workspace_id: str | None,
        assistant_mode_id: str,
        resolved_policy: Any,
    ) -> bool:
        return (
            await self._memory_repository.count_for_context(
                user_id,
                resolved_policy.allowed_scopes,
                workspace_id=workspace_id,
                conversation_id=conversation_id,
                assistant_mode_id=assistant_mode_id,
            )
            == 0
        )
