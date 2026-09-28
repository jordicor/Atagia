"""Synthetic applicability fixtures shared by scorer and provider tests."""

from __future__ import annotations

import json
from pathlib import Path

from atagia.core.config import Settings, default_resource_path
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver
from atagia.models.schemas_memory import (
    ExtractionContextMessage,
    ExtractionConversationContext,
)
from atagia.services.llm_client import (
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
)

MANIFESTS_DIR = Path(__file__).resolve().parents[1] / "src" / "atagia" / "resources" / "manifests"


def _candidate_attr(candidate_tag: str, name: str) -> str | None:
    marker = f'{name}="'
    if marker not in candidate_tag:
        return None
    return candidate_tag.split(marker, 1)[1].split('"', 1)[0]


def _score_keys_by_memory_id(prompt: str) -> dict[str, str]:
    score_keys: dict[str, str] = {}
    for candidate_fragment in prompt.split("<candidate ")[1:]:
        candidate_tag = candidate_fragment.split(">", 1)[0]
        memory_id = _candidate_attr(candidate_tag, "memory_id")
        score_key = _candidate_attr(candidate_tag, "score_key")
        if memory_id is not None and score_key is not None:
            score_keys[memory_id] = score_key
    return score_keys


def _scores_array_from_payload(prompt: str, payload: list[dict[str, object]]) -> list[dict[str, object]]:
    score_keys_by_memory_id = _score_keys_by_memory_id(prompt)
    scores: list[dict[str, object]] = []
    for item in payload:
        score_key = item.get("score_key")
        if score_key is None:
            memory_id = str(item.get("memory_id", ""))
            score_key = score_keys_by_memory_id.get(memory_id, memory_id)
        scores.append(
            {
                "score_key": str(score_key),
                **{
                    key: value
                    for key, value in item.items()
                    if key not in {"memory_id", "score_key"}
                },
            }
        )
    return scores


def _label_for_score(score: object) -> str:
    value = float(score)
    if value <= 0.10:
        return "drop"
    if value <= 0.40:
        return "weak"
    if value <= 0.65:
        return "useful"
    if value <= 0.85:
        return "strong"
    return "exact"


def _card_output_from_payload(prompt: str, payload: list[dict[str, object]]) -> str:
    score_keys_by_memory_id = _score_keys_by_memory_id(prompt)
    lines: list[str] = []
    for item in payload:
        score_key = item.get("score_key")
        if score_key is None:
            memory_id = str(item.get("memory_id", ""))
            score_key = score_keys_by_memory_id.get(memory_id, memory_id)
        label = str(item.get("label") or _label_for_score(item.get("llm_applicability", 0.55)))
        lines.append(f"{score_key} {label}")
    return "\n".join(lines)


def _date_card_output_from_prompt(prompt: str) -> str:
    return "\n".join(
        f"{score_key} none" for score_key in _score_keys_by_memory_id(prompt).values()
    )


class CannedApplicabilityProvider(LLMProvider):
    name = "canned-applicability"

    def __init__(self, payload: list[dict[str, object]]) -> None:
        self.payload = payload
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        purpose = str(request.metadata.get("purpose") or "")
        if purpose == "applicability_relevance_card":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=_card_output_from_payload(
                    request.messages[1].content,
                    self.payload,
                ),
            )
        if purpose == "applicability_date_card":
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=_date_card_output_from_prompt(request.messages[1].content),
            )
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=json.dumps(
                {"scores": _scores_array_from_payload(request.messages[1].content, self.payload)}
            ),
        )

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        raise AssertionError("Embeddings are not used by applicability scorer tests")


def _resolved_policy(mode_id: str = "coding_debug"):
    loader = ManifestLoader(MANIFESTS_DIR)
    manifest = loader.load_all()[mode_id]
    return PolicyResolver().resolve(manifest, None, None)


def _settings() -> Settings:
    return Settings(
        sqlite_path=":memory:",
        migrations_path=default_resource_path("migrations"),
        manifests_path=default_resource_path("manifests"),
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        openai_api_key=None,
        openrouter_api_key=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
        llm_chat_model=None,
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
    )


def _context() -> ExtractionConversationContext:
    return ExtractionConversationContext(
        user_id="usr_1",
        conversation_id="cnv_1",
        source_message_id="msg_1",
        workspace_id=None,
        assistant_mode_id="coding_debug",
        recent_messages=[
            ExtractionContextMessage(role="assistant", content="I previously suggested a retry loop fix."),
            ExtractionContextMessage(role="user", content="It still failed in production."),
        ],
    )


def _candidate(
    memory_id: str,
    *,
    object_type: str = "evidence",
    scope: str = "conversation",
    status: str = "active",
    privacy_level: int = 0,
    valid_from: str | None = None,
    valid_to: str | None = None,
    temporal_type: str = "unknown",
    canonical_text: str = "The websocket retry loop still fails in production.",
    rank: float = 0.5,
    rrf_score: float = 0.05,
    vitality: float = 0.4,
    confirmation_count: int = 0,
    maya_score: float = 0.0,
    updated_at: str = "2026-03-30T21:00:00+00:00",
    retrieval_sources: list[str] | None = None,
) -> dict[str, object]:
    scope_canonical = {
        "conversation": "chat",
        "ephemeral_session": "chat",
        "workspace": "character",
        "global_user": "user",
        "assistant_mode": "legacy_assistant_mode",
    }.get(scope, scope)
    return {
        "id": memory_id,
        "user_id": "usr_1",
        "workspace_id": None,
        "conversation_id": "cnv_1",
        "assistant_mode_id": "coding_debug",
        "user_persona_id": None,
        "platform_id": "default",
        "character_id": None,
        "object_type": object_type,
        "scope": scope,
        "scope_canonical": scope_canonical,
        "canonical_text": canonical_text,
        "payload_json": {"confirmation_count": confirmation_count},
        "source_kind": "extracted",
        "confidence": 0.8,
        "stability": 0.5,
        "vitality": vitality,
        "maya_score": maya_score,
        "privacy_level": privacy_level,
        "sensitivity": "public",
        "platform_locked": 0,
        "platform_id_lock": None,
        "temporal_type": temporal_type,
        "valid_from": valid_from,
        "valid_to": valid_to,
        "status": status,
        "created_at": "2026-03-30T21:00:00+00:00",
        "updated_at": updated_at,
        "rank": rank,
        "rrf_score": rrf_score,
        "retrieval_sources": retrieval_sources or ["fts"],
    }
