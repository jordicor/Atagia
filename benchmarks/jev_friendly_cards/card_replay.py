"""Replay one frozen synthetic candidate through its selected production cards.

This module is loaded by path in each isolated variant process. Its ``atagia``
imports therefore resolve from that process's selected checkout.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
import inspect
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel

from atagia.core.clock import FrozenClock
from atagia.core.source_references import SourceReferenceCatalog
from atagia.memory import extraction_cards
from atagia.memory.intent_classifier import are_claim_key_pairs_equivalent_batch
from atagia.memory.need_detector import (
    NeedDetector,
    _authority_context_from_extraction_context,
)
from atagia.memory.policy_manifest import ManifestLoader, PolicyResolver
from atagia.memory.topic_working_set import TopicWorkingSetUpdater
from atagia.models.schemas_memory import (
    ExplicitLanguagePreference,
    ExtractionConversationContext,
    UserCommunicationProfile,
)
from atagia.services.llm_client import known_intimacy_context_metadata
from atagia.services.model_resolution import (
    component_id_for_llm_purpose,
    examples_enabled_for_component,
    resolve_component_model,
)
from atagia.services.prompt_authority import (
    process_authority_context,
    prompt_authority_metadata,
)


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Replay output requires string dictionary keys")
        return {key: _json_value(item) for key, item in value.items()}
    raise TypeError(f"Unsupported replay output: {type(value).__name__}")


def _card_result(result: extraction_cards.CardResult) -> dict[str, Any]:
    return {
        "parsed": _json_value(result.parsed),
        "raw_output": result.raw_output,
        "malformed_count": result.malformed_count,
    }


def _model(settings: Any, purpose: str) -> str:
    return resolve_component_model(settings, component_id_for_llm_purpose(purpose))


def _policy(settings: Any, mode: str) -> Any:
    manifest = ManifestLoader(Path(settings.manifests_path)).get(mode)
    return PolicyResolver().resolve(manifest, None, None)


def _context(
    case: Mapping[str, Any], fixture: Mapping[str, Any]
) -> ExtractionConversationContext:
    return ExtractionConversationContext.model_validate(
        {
            "user_id": fixture["user_id"],
            "conversation_id": fixture["conversation_id"],
            "source_message_id": fixture["source_message_id"],
            "assistant_mode_id": fixture["assistant_mode_id"],
            "mode": fixture["mode"],
            "privacy_enforcement": fixture["privacy_enforcement"],
            "recent_messages": [
                {"role": message["role"], "content": message["content"]}
                for message in case["context_messages"]
            ],
        }
    )


def _extraction_metadata(
    context: ExtractionConversationContext, policy: Any
) -> dict[str, Any]:
    authority = process_authority_context(
        privacy_enforcement=context.privacy_enforcement,
        user_id=context.user_id,
        privilege_level=context.authenticated_user_privilege_level,
        is_atagia_master=context.authenticated_user_is_atagia_master,
        purpose="memory_extraction",
    )
    return {
        "atagia_partial_stream_retry": "discard_and_retry",
        "atagia_technical_recovery_output_limit_strategy": "caller",
        **prompt_authority_metadata(
            authority, prompt_authority_kind="process_metadata"
        ),
        **(
            known_intimacy_context_metadata(
                reason="resolved_policy_allows_intimacy_context"
            )
            if policy.allow_intimacy_context
            else {}
        ),
    }


def _prior_communication_profile(
    fixture: Mapping[str, Any],
) -> UserCommunicationProfile | None:
    prior = fixture.get("prior_language_profile")
    if prior is None:
        return None
    if not isinstance(prior, dict) or set(prior) != {"response_language"}:
        raise ValueError("Frozen prior language profile must contain response_language")
    # The frozen synthetic fixture is the source of this prior preference; no
    # real message or persisted memory is asserted by this benchmark reference.
    return UserCommunicationProfile(
        explicit_language_preferences=[
            ExplicitLanguagePreference(
                language_code=prior["response_language"],
                preference_kind="default_answer_language",
                context_label="default",
                source_refs=[
                    {
                        "source_kind": "source_message",
                        "source_message_id": (
                            f"synthetic_fixture:{fixture['source_message_id']}:prior_language_profile"
                        ),
                    }
                ],
                confidence=1.0,
            )
        ]
    )


async def run_card_replay(
    client: Any,
    *,
    case: Mapping[str, Any],
    fixture: Mapping[str, Any],
    settings: Any,
) -> dict[str, Any]:
    """Run the production path for one of the seven frozen card families.

    The caller owns the configured client, budget, provider retries, slot, and
    process lifetime. This function never opens a provider or reads environment.
    """

    case_id = case["case_id"]
    family = case["primary_family"]
    if case["origin"] != "synthetic" or family not in {
        "classification",
        "evidence",
        "temporal",
        "language",
        "beliefs",
        "members",
        "topics",
    }:
        raise ValueError("Card replay requires one frozen synthetic family case")
    if fixture["source_message_id"] != case_id:
        raise ValueError("Card fixture does not match the source case")
    candidate_text = fixture["card_replay_candidate"]
    if not isinstance(candidate_text, str) or not candidate_text.strip():
        raise ValueError("Card replay requires a frozen candidate")
    timestamp = datetime.fromisoformat(fixture["occurred_at"])
    if timestamp.utcoffset() is None:
        raise ValueError("Card replay requires an offset-aware frozen clock")

    context = _context(case, fixture)
    policy = _policy(settings, fixture["mode"])
    candidate = extraction_cards.CandidateDraft(
        "cand_001",
        candidate_text,
        kind="belief" if family == "beliefs" else "evidence",
    )
    candidates = (candidate,)
    source = case["source_text"]
    role = case["role"]
    occurred_at = fixture["occurred_at"]
    prior_chunk = fixture["prior_chunk_context"]
    scopes = tuple(fixture["allowed_write_scopes"])
    metadata = _extraction_metadata(context, policy)
    semaphore = asyncio.Semaphore(2)
    extraction_model = resolve_component_model(settings, "extractor")
    examples = examples_enabled_for_component(settings, "extractor")
    common = {
        "message_text": source,
        "role": role,
        "context": context,
        "occurred_at": occurred_at,
        "prior_chunk_context": prior_chunk,
        "candidates": candidates,
        "metadata": metadata,
        "semaphore": semaphore,
    }

    if family == "classification":
        async with asyncio.TaskGroup() as group:
            tasks = {
                card_name: group.create_task(
                    extraction_cards.run_classification_card(
                        client,
                        model=_model(
                            settings,
                            f"memory_extraction_{card_name.removeprefix('memory_')}_card",
                        ),
                        card_name=card_name,
                        allowed_write_scopes=scopes,
                        include_examples=examples,
                        **common,
                    )
                )
                for card_name in ("memory_kind", "memory_scope", "memory_confidence")
            }
        cards = {name: _card_result(task.result()) for name, task in tasks.items()}
        output: dict[str, Any] = {"cards": cards}

    elif family == "evidence":
        arguments: dict[str, Any] = {}
        if (
            "support_model"
            in inspect.signature(extraction_cards.run_evidence_card).parameters
        ):
            arguments = {
                "support_model": _model(
                    settings, "memory_extraction_evidence_support_card"
                ),
                "preserve_model": _model(
                    settings, "memory_extraction_preserve_verbatim_card"
                ),
            }
        result = await extraction_cards.run_evidence_card(
            client,
            model=extraction_model,
            evidence_model=resolve_component_model(settings, "extraction_evidence"),
            resolved_policy=policy,
            allowed_write_scopes=scopes,
            include_examples=examples,
            **arguments,
            **common,
        )
        catalog = SourceReferenceCatalog(source)
        row = result.parsed[candidate.candidate_id]
        reference = (
            catalog.resolve(row["start_ref"], row["end_ref"])
            if row["start_ref"] is not None
            else None
        )
        output = {
            "card": _card_result(result),
            "source_reference": reference.model_dump(mode="json")
            if reference
            else None,
            "source_quote": reference.quote(source) if reference else None,
        }

    elif family == "temporal":
        arguments = {}
        if (
            "temporal_type_model"
            in inspect.signature(extraction_cards.run_temporal_cards).parameters
        ):
            arguments["temporal_type_model"] = _model(
                settings, "memory_extraction_temporal_type_card"
            )
        result = await extraction_cards.run_temporal_cards(
            llm_client=client, model=extraction_model,
            date_model=resolve_component_model(settings, "date_resolution"),
            **arguments, **common
        )
        output = {"card": _card_result(result)}

    elif family == "language":
        detector = NeedDetector(client, FrozenClock(timestamp), settings=settings)
        attempts: dict[str, Any] = {}
        prior_profile = _prior_communication_profile(fixture)
        authority = _authority_context_from_extraction_context(
            context, purpose="need_detection"
        )
        language_args = {
            "message_text": source,
            "role": role,
            "context": context,
            "resolved_policy": policy,
            "content_language_profile": [],
            "user_communication_profile": prior_profile,
            "prompt_authority_context": authority,
            "card_attempts": attempts,
        }
        query = await detector._run_card(card_name="query_language", **language_args)
        if not query.parse_valid:
            raise ValueError(
                f"Invalid query-language card: {query.error or query.raw_output}"
            )
        answer = await detector._run_card(
            card_name="answer_language",
            query_language=query.parsed["query_language"],
            **language_args,
        )
        if not answer.parse_valid:
            raise ValueError(
                f"Invalid answer-language card: {answer.error or answer.raw_output}"
            )
        output = {
            "cards": {
                name: {
                    "parsed": _json_value(call.parsed),
                    "raw_output": call.raw_output,
                }
                for name, call in attempts.items()
            },
            "prior_language_profile_input": fixture.get("prior_language_profile"),
            "prior_language_profile_applied": prior_profile is not None,
        }

    elif family == "beliefs":
        result = await extraction_cards.run_belief_cards(
            client, model=extraction_model, include_examples=examples, **common
        )
        comparator_key = fixture["belief"]["candidate_key"]
        catalog_keys = fixture["belief"]["catalog_keys"]
        equivalence = (
            await are_claim_key_pairs_equivalent_batch(
                client,
                resolve_component_model(settings, "intent_classifier"),
                [(comparator_key, key) for key in catalog_keys],
                user_id=context.user_id,
            )
            if catalog_keys else []
        )
        output = {
            "key_generation": _card_result(result),
            "frozen_key_comparator": {
                "candidate_key": comparator_key,
                "catalog_equivalence": dict(
                    zip(catalog_keys, equivalence, strict=True)
                ),
            },
        }

    elif family == "members":
        arguments = {}
        if (
            "identity_model"
            in inspect.signature(extraction_cards.run_coverage_members_card).parameters
        ):
            arguments["identity_model"] = _model(
                settings, "memory_extraction_coverage_member_identity_card"
            )
        result = await extraction_cards.run_coverage_members_card(
            client,
            model=extraction_model,
            include_examples=examples,
            **arguments,
            **common,
        )
        output = {
            "card": _card_result(result),
            "identity_basis": "generated_member_list",
        }

    else:
        updater = TopicWorkingSetUpdater(
            llm_client=client,
            clock=FrozenClock(timestamp),
            topic_repository=None,
            message_repository=None,
            settings=settings,
        )
        plan = await updater._plan_updates(
            user_id=context.user_id,
            conversation_id=context.conversation_id,
            snapshot=fixture["topic"]["snapshot"],
            messages=[
                {
                    "id": context.source_message_id,
                    "role": role,
                    "text": source,
                    "created_at": occurred_at,
                }
            ],
        )
        output = {"plan": plan.model_dump(mode="json")}

    return {
        "case_id": case_id,
        "family": family,
        "candidate_id": candidate.candidate_id,
        **_json_value(output),
    }
