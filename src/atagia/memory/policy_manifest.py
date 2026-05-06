"""Retrieval profile loading, resolution, and legacy database syncing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import aiosqlite
from pydantic import BaseModel, ConfigDict, Field, field_validator

from atagia.core import json_utils
from atagia.core.clock import Clock
from atagia.core.canonical import canonical_json_bytes, canonical_json_hash
from atagia.models.schemas_memory import (
    RetrievalProfileId,
    RetrievalProfileManifest,
    ContextCachePolicy,
    MemoryObjectType,
    MemoryScope,
    NeedTrigger,
    OperationalPolicyOverride,
    RetrievalParams,
)

EXPECTED_RETRIEVAL_PROFILE_IDS = frozenset(profile.value for profile in RetrievalProfileId)
DEFAULT_RETRIEVAL_SCOPE_FILTER = [
    MemoryScope.GLOBAL_USER,
    MemoryScope.WORKSPACE,
    MemoryScope.CONVERSATION,
    MemoryScope.EPHEMERAL_SESSION,
]


def _ensure_unique_values[T](values: list[T]) -> list[T]:
    if len(values) != len(set(values)):
        raise ValueError("List values must be unique")
    return values


def _manifest_payload(manifest: RetrievalProfileManifest) -> dict[str, Any]:
    return manifest.model_dump(mode="json")


def _manifest_json(manifest: RetrievalProfileManifest) -> str:
    return canonical_json_bytes(_manifest_payload(manifest)).decode("utf-8")


def compute_prompt_hash(payload: dict[str, Any]) -> str:
    """Compute a stable manifest hash from canonical JSON bytes."""
    return canonical_json_hash(payload)


class RetrievalParamsOverride(BaseModel):
    """Optional override for retrieval parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fts_limit: int | None = Field(default=None, ge=0)
    vector_limit: int | None = Field(default=None, ge=0)
    graph_hops: int | None = Field(default=None, ge=0)
    rerank_top_k: int | None = Field(default=None, gt=0)
    final_context_items: int | None = Field(default=None, gt=0)


class PolicyOverride(BaseModel):
    """Workspace or conversation-level retrieval-profile tuning overrides."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    allow_intimacy_context: bool | None = None
    preferred_memory_types: list[MemoryObjectType] | None = None
    need_triggers: list[NeedTrigger] | None = None
    contract_dimensions_priority: list[str] | None = None
    privacy_ceiling: int | None = Field(default=None, ge=0, le=3)
    context_budget_tokens: int | None = Field(default=None, gt=0)
    transcript_budget_tokens: int | None = Field(default=None, gt=0)
    retrieval_params: RetrievalParamsOverride | None = None

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


class ResolvedRetrievalPolicy(BaseModel):
    """Fully resolved retrieval profile ready for retrieval planning."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: RetrievalProfileId
    display_name: str
    prompt_hash: str
    cross_chat_allowed: bool
    allow_intimacy_context: bool
    allowed_scopes: list[MemoryScope]
    preferred_memory_types: list[MemoryObjectType]
    need_triggers: list[NeedTrigger]
    contract_dimensions_priority: list[str]
    privacy_ceiling: int
    context_budget_tokens: int
    transcript_budget_tokens: int
    retrieval_params: RetrievalParams
    context_cache_policy: ContextCachePolicy


class ManifestLoader:
    """Filesystem-backed retrieval profile manifest loader."""

    def __init__(self, manifests_dir: Path) -> None:
        self._manifests_dir = manifests_dir
        self._manifests: dict[str, RetrievalProfileManifest] = {}

    def load_all(self) -> dict[str, RetrievalProfileManifest]:
        if not self._manifests_dir.exists():
            raise FileNotFoundError(f"Missing manifests directory: {self._manifests_dir}")

        manifests: dict[str, RetrievalProfileManifest] = {}
        for path in sorted(self._manifests_dir.glob("*.json")):
            payload = json_utils.loads(path.read_text(encoding="utf-8"))
            manifest = RetrievalProfileManifest.model_validate(payload)
            manifest = manifest.model_copy(update={"prompt_hash": compute_prompt_hash(_manifest_payload(manifest))})
            profile_id = manifest.profile_id.value
            if profile_id in manifests:
                raise ValueError(f"Duplicate retrieval profile manifest: {profile_id}")
            manifests[profile_id] = manifest

        missing = EXPECTED_RETRIEVAL_PROFILE_IDS - manifests.keys()
        if missing:
            missing_profiles = ", ".join(sorted(missing))
            raise ValueError(f"Missing retrieval profile manifests: {missing_profiles}")

        self._manifests = manifests
        return dict(manifests)

    def get(self, mode_id: str) -> RetrievalProfileManifest:
        if not self._manifests:
            self.load_all()
        return self._manifests[mode_id]


class PolicyResolver:
    """Merge retrieval profile tuning with runtime overrides."""

    def resolve(
        self,
        manifest: RetrievalProfileManifest,
        workspace_override: dict[str, Any] | None,
        conversation_override: dict[str, Any] | None,
        operational_override: OperationalPolicyOverride | dict[str, Any] | None = None,
    ) -> ResolvedRetrievalPolicy:
        workspace = PolicyOverride.model_validate(workspace_override or {})
        conversation = PolicyOverride.model_validate(conversation_override or {})
        operational = OperationalPolicyOverride.model_validate(operational_override or {})
        base_retrieval_params = self._resolve_retrieval_params(
            manifest.retrieval_params,
            workspace.retrieval_params,
            conversation.retrieval_params,
        )

        return ResolvedRetrievalPolicy(
            profile_id=manifest.profile_id,
            display_name=manifest.display_name,
            prompt_hash=manifest.prompt_hash or compute_prompt_hash(_manifest_payload(manifest)),
            cross_chat_allowed=True,
            allow_intimacy_context=self._resolve_allow_intimacy_context(
                manifest.allow_intimacy_context,
                workspace.allow_intimacy_context,
                conversation.allow_intimacy_context,
            ),
            allowed_scopes=list(DEFAULT_RETRIEVAL_SCOPE_FILTER),
            preferred_memory_types=self._pick_most_specific_list(
                manifest.preferred_memory_types,
                workspace.preferred_memory_types,
                conversation.preferred_memory_types,
                operational.preferred_memory_types,
            ),
            need_triggers=self._resolve_need_triggers(
                self._pick_most_specific_list(
                    manifest.need_triggers,
                    workspace.need_triggers,
                    conversation.need_triggers,
                ),
                operational.need_triggers,
            ),
            contract_dimensions_priority=self._pick_most_specific_list(
                manifest.contract_dimensions_priority,
                workspace.contract_dimensions_priority,
                conversation.contract_dimensions_priority,
                operational.contract_dimensions_priority,
            ),
            privacy_ceiling=self._resolve_privacy_ceiling(
                manifest.privacy_ceiling,
                workspace.privacy_ceiling,
                conversation.privacy_ceiling,
            ),
            context_budget_tokens=self._resolve_restricted_scalar(
                self._pick_most_specific_scalar(
                    manifest.context_budget_tokens,
                    workspace.context_budget_tokens,
                    conversation.context_budget_tokens,
                ),
                operational.context_budget_tokens,
            ),
            transcript_budget_tokens=self._resolve_restricted_scalar(
                self._pick_most_specific_scalar(
                    manifest.transcript_budget_tokens,
                    workspace.transcript_budget_tokens,
                    conversation.transcript_budget_tokens,
                ),
                operational.transcript_budget_tokens,
            ),
            retrieval_params=self._resolve_operational_retrieval_params(
                base_retrieval_params,
                operational.retrieval_params,
            ),
            context_cache_policy=manifest.context_cache_policy,
        )

    @staticmethod
    def _resolve_allow_intimacy_context(
        manifest_value: bool,
        workspace_value: bool | None,
        conversation_value: bool | None,
    ) -> bool:
        return manifest_value and (workspace_value is not False) and (conversation_value is not False)

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
        operational_value: list[T] | None = None,
    ) -> list[T]:
        resolved = PolicyResolver._pick_most_specific_scalar(
            manifest_value,
            workspace_value,
            conversation_value,
        )
        return list(operational_value if operational_value is not None else resolved)

    @staticmethod
    def _resolve_need_triggers(
        resolved_value: list[NeedTrigger],
        operational_value: list[NeedTrigger] | None,
    ) -> list[NeedTrigger]:
        if operational_value is None:
            return list(resolved_value)
        merged: list[NeedTrigger] = []
        for trigger in [*resolved_value, *operational_value]:
            if trigger in merged:
                continue
            merged.append(trigger)
        return merged

    @staticmethod
    def _resolve_restricted_scalar[T](resolved_value: T, operational_value: T | None) -> T:
        if operational_value is None:
            return resolved_value
        return min(resolved_value, operational_value)

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

    @staticmethod
    def _resolve_operational_retrieval_params(
        resolved_value: RetrievalParams,
        operational_value: Any | None,
    ) -> RetrievalParams:
        if operational_value is None:
            return resolved_value
        resolved = resolved_value.model_dump()
        for field_name in RetrievalParams.model_fields:
            override_value = getattr(operational_value, field_name)
            if override_value is not None:
                resolved[field_name] = min(resolved[field_name], override_value)
        return RetrievalParams.model_validate(resolved)


def compute_effective_policy_hash(resolved: ResolvedRetrievalPolicy) -> str:
    """Compute a stable hash for the fully resolved runtime policy."""
    return canonical_json_hash(resolved.model_dump(mode="json", exclude={"prompt_hash"}))


async def sync_assistant_modes(
    connection: aiosqlite.Connection,
    manifests: dict[str, RetrievalProfileManifest],
    clock: Clock,
) -> None:
    """Upsert legacy assistant_modes rows so the database matches profiles."""
    missing = EXPECTED_RETRIEVAL_PROFILE_IDS - manifests.keys()
    if missing:
        missing_profiles = ", ".join(sorted(missing))
        raise ValueError(f"Cannot sync retrieval profiles, missing manifests: {missing_profiles}")

    timestamp = clock.now().isoformat()
    for mode_id in sorted(EXPECTED_RETRIEVAL_PROFILE_IDS):
        manifest = manifests[mode_id]
        await connection.execute(
            """
            INSERT INTO assistant_modes(
                id,
                display_name,
                prompt_hash,
                memory_policy_json,
                privacy_ceiling,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                display_name = excluded.display_name,
                prompt_hash = excluded.prompt_hash,
                memory_policy_json = excluded.memory_policy_json,
                privacy_ceiling = excluded.privacy_ceiling,
                updated_at = excluded.updated_at
            """,
            (
                mode_id,
                manifest.display_name,
                manifest.prompt_hash or compute_prompt_hash(_manifest_payload(manifest)),
                _manifest_json(manifest),
                int(manifest.privacy_ceiling),
                timestamp,
                timestamp,
            ),
        )
    await connection.commit()


async def load_and_sync_assistant_modes(
    connection: aiosqlite.Connection,
    manifests_dir: Path,
    clock: Clock,
) -> dict[str, RetrievalProfileManifest]:
    """Load manifests from disk and persist them into assistant_modes."""
    loader = ManifestLoader(manifests_dir)
    manifests = loader.load_all()
    await sync_assistant_modes(connection, manifests, clock)
    return manifests
