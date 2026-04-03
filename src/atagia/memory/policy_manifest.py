"""Assistant mode manifest loading, resolution, and database syncing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import aiosqlite
from pydantic import BaseModel, ConfigDict, Field, field_validator

from atagia.core.clock import Clock
from atagia.models.schemas_memory import (
    AssistantModeId,
    AssistantModeManifest,
    ContextCachePolicy,
    MemoryObjectType,
    MemoryScope,
    NeedTrigger,
    RetrievalParams,
)

EXPECTED_ASSISTANT_MODE_IDS = frozenset(mode.value for mode in AssistantModeId)


def _ensure_unique_values[T](values: list[T]) -> list[T]:
    if len(values) != len(set(values)):
        raise ValueError("List values must be unique")
    return values


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _manifest_payload(manifest: AssistantModeManifest) -> dict[str, Any]:
    return manifest.model_dump(mode="json")


def _manifest_json(manifest: AssistantModeManifest) -> str:
    return _canonical_json_bytes(_manifest_payload(manifest)).decode("utf-8")


def compute_prompt_hash(payload: dict[str, Any]) -> str:
    """Compute a stable manifest hash from canonical JSON bytes."""
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


class RetrievalParamsOverride(BaseModel):
    """Optional override for retrieval parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fts_limit: int | None = Field(default=None, ge=0)
    vector_limit: int | None = Field(default=None, ge=0)
    graph_hops: int | None = Field(default=None, ge=0)
    rerank_top_k: int | None = Field(default=None, gt=0)
    final_context_items: int | None = Field(default=None, gt=0)


class PolicyOverride(BaseModel):
    """Workspace or conversation-level policy overrides."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cross_chat_allowed: bool | None = None
    allowed_scopes: list[MemoryScope] | None = None
    preferred_memory_types: list[MemoryObjectType] | None = None
    need_triggers: list[NeedTrigger] | None = None
    contract_dimensions_priority: list[str] | None = None
    privacy_ceiling: int | None = Field(default=None, ge=0, le=3)
    context_budget_tokens: int | None = Field(default=None, gt=0)
    retrieval_params: RetrievalParamsOverride | None = None

    @field_validator("allowed_scopes")
    @classmethod
    def validate_allowed_scopes(cls, values: list[MemoryScope] | None) -> list[MemoryScope] | None:
        if values is None:
            return values
        return _ensure_unique_values(values)

    @field_validator("preferred_memory_types")
    @classmethod
    def validate_preferred_memory_types(
        cls,
        values: list[MemoryObjectType] | None,
    ) -> list[MemoryObjectType] | None:
        if values is None:
            return values
        return _ensure_unique_values(values)

    @field_validator("need_triggers")
    @classmethod
    def validate_need_triggers(cls, values: list[NeedTrigger] | None) -> list[NeedTrigger] | None:
        if values is None:
            return values
        return _ensure_unique_values(values)

    @field_validator("contract_dimensions_priority")
    @classmethod
    def validate_contract_dimensions_priority(cls, values: list[str] | None) -> list[str] | None:
        if values is None:
            return values
        if any(not value.strip() for value in values):
            raise ValueError("Contract dimensions must be non-empty strings")
        return _ensure_unique_values(values)


class ResolvedPolicy(BaseModel):
    """Fully resolved policy ready for retrieval planning."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    assistant_mode_id: AssistantModeId
    display_name: str
    prompt_hash: str
    cross_chat_allowed: bool
    allowed_scopes: list[MemoryScope]
    preferred_memory_types: list[MemoryObjectType]
    need_triggers: list[NeedTrigger]
    contract_dimensions_priority: list[str]
    privacy_ceiling: int
    context_budget_tokens: int
    retrieval_params: RetrievalParams
    context_cache_policy: ContextCachePolicy


class ManifestLoader:
    """Filesystem-backed assistant mode manifest loader."""

    def __init__(self, manifests_dir: Path) -> None:
        self._manifests_dir = manifests_dir
        self._manifests: dict[str, AssistantModeManifest] = {}

    def load_all(self) -> dict[str, AssistantModeManifest]:
        if not self._manifests_dir.exists():
            raise FileNotFoundError(f"Missing manifests directory: {self._manifests_dir}")

        manifests: dict[str, AssistantModeManifest] = {}
        for path in sorted(self._manifests_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            manifest = AssistantModeManifest.model_validate(payload)
            manifest = manifest.model_copy(update={"prompt_hash": compute_prompt_hash(_manifest_payload(manifest))})
            mode_id = manifest.assistant_mode_id.value
            if mode_id in manifests:
                raise ValueError(f"Duplicate manifest for assistant mode: {mode_id}")
            manifests[mode_id] = manifest

        missing = EXPECTED_ASSISTANT_MODE_IDS - manifests.keys()
        if missing:
            missing_modes = ", ".join(sorted(missing))
            raise ValueError(f"Missing assistant mode manifests: {missing_modes}")

        self._manifests = manifests
        return dict(manifests)

    def get(self, mode_id: str) -> AssistantModeManifest:
        if not self._manifests:
            self.load_all()
        return self._manifests[mode_id]


class PolicyResolver:
    """Merge assistant mode policies with workspace and conversation overrides."""

    def resolve(
        self,
        manifest: AssistantModeManifest,
        workspace_override: dict[str, Any] | None,
        conversation_override: dict[str, Any] | None,
    ) -> ResolvedPolicy:
        workspace = PolicyOverride.model_validate(workspace_override or {})
        conversation = PolicyOverride.model_validate(conversation_override or {})

        return ResolvedPolicy(
            assistant_mode_id=manifest.assistant_mode_id,
            display_name=manifest.display_name,
            prompt_hash=manifest.prompt_hash or compute_prompt_hash(_manifest_payload(manifest)),
            cross_chat_allowed=self._resolve_cross_chat_allowed(
                manifest.cross_chat_allowed,
                workspace.cross_chat_allowed,
                conversation.cross_chat_allowed,
            ),
            allowed_scopes=self._resolve_allowed_scopes(
                manifest.allowed_scopes,
                workspace.allowed_scopes,
                conversation.allowed_scopes,
            ),
            preferred_memory_types=self._pick_most_specific_list(
                manifest.preferred_memory_types,
                workspace.preferred_memory_types,
                conversation.preferred_memory_types,
            ),
            need_triggers=self._pick_most_specific_list(
                manifest.need_triggers,
                workspace.need_triggers,
                conversation.need_triggers,
            ),
            contract_dimensions_priority=self._pick_most_specific_list(
                manifest.contract_dimensions_priority,
                workspace.contract_dimensions_priority,
                conversation.contract_dimensions_priority,
            ),
            privacy_ceiling=self._resolve_privacy_ceiling(
                manifest.privacy_ceiling,
                workspace.privacy_ceiling,
                conversation.privacy_ceiling,
            ),
            context_budget_tokens=self._pick_most_specific_scalar(
                manifest.context_budget_tokens,
                workspace.context_budget_tokens,
                conversation.context_budget_tokens,
            ),
            retrieval_params=self._resolve_retrieval_params(
                manifest.retrieval_params,
                workspace.retrieval_params,
                conversation.retrieval_params,
            ),
            context_cache_policy=manifest.context_cache_policy,
        )

    @staticmethod
    def _resolve_cross_chat_allowed(
        manifest_value: bool,
        workspace_value: bool | None,
        conversation_value: bool | None,
    ) -> bool:
        return manifest_value and (workspace_value is not False) and (conversation_value is not False)

    @staticmethod
    def _resolve_allowed_scopes(
        manifest_scopes: list[MemoryScope],
        workspace_scopes: list[MemoryScope] | None,
        conversation_scopes: list[MemoryScope] | None,
    ) -> list[MemoryScope]:
        allowed = list(manifest_scopes)
        for override_scopes in (workspace_scopes, conversation_scopes):
            if override_scopes is None:
                continue
            override_set = set(override_scopes)
            allowed = [scope for scope in allowed if scope in override_set]
        return allowed

    @staticmethod
    def _resolve_privacy_ceiling(
        manifest_value: int,
        workspace_value: int | None,
        conversation_value: int | None,
    ) -> int:
        resolved = manifest_value
        for value in (workspace_value, conversation_value):
            if value is not None:
                resolved = min(resolved, value)
        return resolved

    @staticmethod
    def _pick_most_specific_scalar[T](
        manifest_value: T,
        workspace_value: T | None,
        conversation_value: T | None,
    ) -> T:
        if conversation_value is not None:
            return conversation_value
        if workspace_value is not None:
            return workspace_value
        return manifest_value

    @staticmethod
    def _pick_most_specific_list[T](
        manifest_value: list[T],
        workspace_value: list[T] | None,
        conversation_value: list[T] | None,
    ) -> list[T]:
        return list(
            PolicyResolver._pick_most_specific_scalar(
                manifest_value,
                workspace_value,
                conversation_value,
            )
        )

    @staticmethod
    def _resolve_retrieval_params(
        manifest_value: RetrievalParams,
        workspace_value: RetrievalParamsOverride | None,
        conversation_value: RetrievalParamsOverride | None,
    ) -> RetrievalParams:
        resolved = manifest_value.model_dump()
        for field_name in RetrievalParams.model_fields:
            for override in (workspace_value, conversation_value):
                if override is None:
                    continue
                override_value = getattr(override, field_name)
                if override_value is not None:
                    resolved[field_name] = override_value
        return RetrievalParams.model_validate(resolved)


async def sync_assistant_modes(
    connection: aiosqlite.Connection,
    manifests: dict[str, AssistantModeManifest],
    clock: Clock,
) -> None:
    """Upsert assistant mode rows so the database matches manifest files."""
    missing = EXPECTED_ASSISTANT_MODE_IDS - manifests.keys()
    if missing:
        missing_modes = ", ".join(sorted(missing))
        raise ValueError(f"Cannot sync assistant modes, missing manifests: {missing_modes}")

    timestamp = clock.now().isoformat()
    for mode_id in sorted(EXPECTED_ASSISTANT_MODE_IDS):
        manifest = manifests[mode_id]
        await connection.execute(
            """
            INSERT INTO assistant_modes(id, display_name, prompt_hash, memory_policy_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                display_name = excluded.display_name,
                prompt_hash = excluded.prompt_hash,
                memory_policy_json = excluded.memory_policy_json,
                updated_at = excluded.updated_at
            """,
            (
                mode_id,
                manifest.display_name,
                manifest.prompt_hash or compute_prompt_hash(_manifest_payload(manifest)),
                _manifest_json(manifest),
                timestamp,
                timestamp,
            ),
        )
    await connection.commit()


async def load_and_sync_assistant_modes(
    connection: aiosqlite.Connection,
    manifests_dir: Path,
    clock: Clock,
) -> dict[str, AssistantModeManifest]:
    """Load manifests from disk and persist them into assistant_modes."""
    loader = ManifestLoader(manifests_dir)
    manifests = loader.load_all()
    await sync_assistant_modes(connection, manifests, clock)
    return manifests
