"""Tests for operational policy overlays."""

from __future__ import annotations

from pathlib import Path

from atagia.memory.policy_manifest import (
    ManifestLoader,
    PolicyResolver,
    PolicyOverride,
    compute_effective_policy_hash,
)
from atagia.models.schemas_memory import (
    MemoryObjectType,
    MemoryScope,
    NeedTrigger,
    OperationalPolicyOverride,
    OperationalRetrievalParamsOverride,
)

MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


def test_operational_overlay_restricts_without_mutating_prompt_hash() -> None:
    manifest = ManifestLoader(MANIFESTS_DIR).load_all()["coding_debug"]
    resolver = PolicyResolver()

    resolved = resolver.resolve(
        manifest,
        workspace_override={
            "allowed_scopes": ["workspace", "conversation"],
            "preferred_memory_types": ["evidence"],
            "privacy_ceiling": 0,
            "context_budget_tokens": 4000,
            "retrieval_params": {"fts_limit": 20, "final_context_items": 6},
        },
        conversation_override={
            "preferred_memory_types": ["belief"],
            "context_budget_tokens": 3000,
            "retrieval_params": {"fts_limit": 12, "rerank_top_k": 10},
        },
        operational_override=OperationalPolicyOverride(
            allowed_scopes=[MemoryScope.CONVERSATION],
            preferred_memory_types=[MemoryObjectType.CONSEQUENCE_CHAIN],
            need_triggers=[NeedTrigger.SENSITIVE_CONTEXT],
            contract_dimensions_priority=["directness"],
            context_budget_tokens=1200,
            transcript_budget_tokens=2000,
            retrieval_params=OperationalRetrievalParamsOverride(
                fts_limit=5,
                vector_limit=3,
                rerank_top_k=4,
                final_context_items=2,
            ),
        ),
    )

    assert resolved.prompt_hash == manifest.prompt_hash
    assert compute_effective_policy_hash(resolved) != compute_effective_policy_hash(
        resolver.resolve(manifest, None, None)
    )
    assert resolved.allowed_scopes == [MemoryScope.CONVERSATION]
    assert resolved.preferred_memory_types == [MemoryObjectType.CONSEQUENCE_CHAIN]
    assert resolved.need_triggers[-1] is NeedTrigger.SENSITIVE_CONTEXT
    assert len(resolved.need_triggers) == len(set(resolved.need_triggers))
    assert resolved.contract_dimensions_priority == ["directness"]
    assert resolved.privacy_ceiling == 0
    assert resolved.context_budget_tokens == 1200
    assert resolved.transcript_budget_tokens == 2000
    assert resolved.retrieval_params.fts_limit == 5
    assert resolved.retrieval_params.vector_limit == 3
    assert resolved.retrieval_params.rerank_top_k == 4
    assert resolved.retrieval_params.final_context_items == 2


def test_operational_override_dict_is_accepted_by_policy_models() -> None:
    override = OperationalPolicyOverride(
        allowed_scopes=[MemoryScope.CONVERSATION],
        preferred_memory_types=[MemoryObjectType.EVIDENCE],
        need_triggers=[NeedTrigger.AMBIGUITY],
        context_budget_tokens=1000,
    )

    payload = override.to_policy_override_dict()

    assert payload == {
        "allowed_scopes": ["conversation"],
        "preferred_memory_types": ["evidence"],
        "need_triggers": ["ambiguity"],
        "context_budget_tokens": 1000,
    }
    assert PolicyOverride.model_validate(payload).context_budget_tokens == 1000
