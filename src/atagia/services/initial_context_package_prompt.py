"""Prompt-time reads for prepared initial-context packages."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping

import aiosqlite

from atagia.core.clock import Clock
from atagia.core.initial_context_package_repository import InitialContextPackageRepository
from atagia.memory.context_envelope import ContextEnvelopeBudget
from atagia.memory.policy_manifest import ResolvedRetrievalPolicy
from atagia.models.schemas_initial_context_package import (
    InitialContextPackageBuildStatus,
    InitialContextPackageKind,
    InitialContextPackageProfileItem,
    InitialContextPackageRecord,
    initial_context_package_key_hash,
)
from atagia.models.schemas_memory import OperationalProfileSnapshot
from atagia.services.chat_support import estimate_tokens
from atagia.services.initial_context_package_builder import (
    INITIAL_CONTEXT_PACKAGE_SCHEMA_VERSION,
)
from atagia.services.initial_context_package_keys import (
    build_initial_context_package_key,
    initial_context_package_subject,
)
from atagia.services.initial_context_package_signatures import (
    build_initial_context_package_coordinate_signature,
    build_initial_context_package_policy_signature,
)
from atagia.services.prompt_authority import PromptAuthorityContext

INITIAL_CONTEXT_PACKAGE_PROMPT_MAX_TOKENS = 900


@dataclass(frozen=True, slots=True)
class InitialContextPackagePromptAssembly:
    """Rendered package block plus request-safe diagnostics."""

    block: str
    diagnostics: dict[str, Any]


def drop_initial_context_package_for_overflow(
    assembly: InitialContextPackagePromptAssembly,
) -> InitialContextPackagePromptAssembly:
    """Return diagnostics for dropping prepared context below live evidence."""

    diagnostics = dict(assembly.diagnostics)
    dropped_sections = list(diagnostics.get("dropped_sections") or [])
    dropped_sections.append("prepared_initial_context:overflow_precedence")
    diagnostics.update(
        {
            "rendered": False,
            "tokens_estimate": 0,
            "overflow_dropped": True,
            "dropped_sections": dropped_sections,
        }
    )
    return InitialContextPackagePromptAssembly("", diagnostics)


async def assemble_initial_context_package_prompt(
    connection: aiosqlite.Connection,
    clock: Clock,
    *,
    enabled: bool,
    user_id: str,
    conversation_id: str,
    conversation: Mapping[str, Any],
    resolved_policy: ResolvedRetrievalPolicy,
    authority_context: PromptAuthorityContext,
    operational_profile: OperationalProfileSnapshot | None,
    context_envelope_budget: ContextEnvelopeBudget,
    retrieved_context_tokens: int,
    selected_memory_ids: list[str],
    topic_context_block: str,
    live_contract_block: str,
    live_state_block: str,
    recent_transcript_message_ids: set[str],
    include_recent_verbatim_seed: bool = True,
    prompt_budget_tokens: int = INITIAL_CONTEXT_PACKAGE_PROMPT_MAX_TOKENS,
) -> InitialContextPackagePromptAssembly:
    """Read and render prepared packages without calling an LLM."""

    started = perf_counter()
    base_diagnostics: dict[str, Any] = {
        "enabled": bool(enabled),
        "rendered": False,
        "read_ms": 0.0,
        "budget_tokens": 0,
        "tokens_estimate": 0,
        "packages": [],
        "selected_profile_items": 0,
        "dropped_profile_items": 0,
        "selected_curated_items": 0,
        "dropped_curated_items": 0,
        "deduped_source_refs": 0,
        "dropped_sections": [],
        "known_empty": {},
    }
    if not enabled:
        base_diagnostics["read_ms"] = _elapsed_ms(started)
        return InitialContextPackagePromptAssembly("", base_diagnostics)
    if _authority_requires_live_context(authority_context):
        base_diagnostics["disabled_reason"] = "authority_requires_live_context"
        base_diagnostics["read_ms"] = _elapsed_ms(started)
        return InitialContextPackagePromptAssembly("", base_diagnostics)

    available_budget = _available_package_budget(
        context_envelope_budget,
        retrieved_context_tokens=retrieved_context_tokens,
        topic_context_block=topic_context_block,
        prompt_budget_tokens=prompt_budget_tokens,
    )
    base_diagnostics["budget_tokens"] = available_budget
    if available_budget <= 0:
        base_diagnostics["disabled_reason"] = "no_package_budget"
        base_diagnostics["read_ms"] = _elapsed_ms(started)
        return InitialContextPackagePromptAssembly("", base_diagnostics)

    repository = InitialContextPackageRepository(connection, clock)
    expected = await _expected_packages(
        connection,
        user_id=user_id,
        conversation_id=conversation_id,
        conversation=conversation,
        resolved_policy=resolved_policy,
        privacy_enforcement=authority_context.effective_privacy_enforcement,
        operational_profile=operational_profile,
    )
    read_results: list[_PackageRead] = []
    for request in expected:
        read_results.append(await _read_expected_package(repository, request))

    active_packages = [result.package for result in read_results if result.package is not None]
    diagnostics = {
        **base_diagnostics,
        "packages": [result.diagnostics for result in read_results],
    }
    if not active_packages:
        diagnostics["read_ms"] = _elapsed_ms(started)
        return InitialContextPackagePromptAssembly("", diagnostics)

    selected_source_keys = {("memory", memory_id) for memory_id in selected_memory_ids}
    selected_source_keys.update(
        ("message", message_id) for message_id in recent_transcript_message_ids
    )
    render_result = _render_packages(
        active_packages,
        selected_source_keys=selected_source_keys,
        live_topic_present=bool(topic_context_block.strip()),
        live_contract_present=bool(live_contract_block.strip()),
        live_state_present=bool(live_state_block.strip()),
        include_recent_verbatim_seed=include_recent_verbatim_seed,
        budget_tokens=available_budget,
    )
    diagnostics.update(render_result.diagnostics)
    diagnostics["read_ms"] = _elapsed_ms(started)
    return InitialContextPackagePromptAssembly(
        render_result.block,
        diagnostics,
    )


@dataclass(frozen=True, slots=True)
class _ExpectedPackage:
    kind: InitialContextPackageKind
    user_id: str
    key_hash: str
    subject_json: dict[str, Any]
    retrieval_profile_id: str
    conversation_id: str | None


@dataclass(frozen=True, slots=True)
class _PackageRead:
    package: InitialContextPackageRecord | None
    diagnostics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _RenderResult:
    block: str
    diagnostics: dict[str, Any]


async def _expected_packages(
    connection: aiosqlite.Connection,
    *,
    user_id: str,
    conversation_id: str,
    conversation: Mapping[str, Any],
    resolved_policy: ResolvedRetrievalPolicy,
    privacy_enforcement: str,
    operational_profile: OperationalProfileSnapshot | None,
) -> list[_ExpectedPackage]:
    retrieval_profile_id = resolved_policy.profile_id.value
    policy_signature = build_initial_context_package_policy_signature(
        resolved_policy,
        privacy_enforcement=privacy_enforcement,
        authority_context=None,
        operational_profile=operational_profile,
    )
    baseline_subject = _baseline_subject(conversation, retrieval_profile_id)
    baseline_coordinate_signature = await build_initial_context_package_coordinate_signature(
        connection,
        user_id=user_id,
        retrieval_profile_id=retrieval_profile_id,
        conversation_id=None,
        conversation=None,
        user_persona_id=baseline_subject.get("user_persona_id"),
        platform_id=baseline_subject.get("platform_id"),
        character_id=baseline_subject.get("character_id"),
        workspace_id=baseline_subject.get("workspace_id"),
        assistant_mode_id=retrieval_profile_id,
        active_presence_id=_optional_text(conversation.get("active_presence_id")),
        active_space_id=_optional_text(conversation.get("active_space_id")),
        active_mind_id=_optional_text(conversation.get("active_mind_id")),
        mind_topology=_optional_text(conversation.get("mind_topology")),
        active_embodiment_id=_optional_text(conversation.get("active_embodiment_id")),
        active_realm_id=_optional_text(conversation.get("active_realm_id")),
        incognito=(
            _coerce_bool(conversation.get("incognito"))
            or _coerce_bool(conversation.get("isolated_mode"))
        ),
    )
    baseline_key = build_initial_context_package_key(
        version=INITIAL_CONTEXT_PACKAGE_SCHEMA_VERSION,
        package_kind=InitialContextPackageKind.BASELINE,
        user_id=user_id,
        conversation_id=None,
        retrieval_profile_id=retrieval_profile_id,
        subject_json=baseline_subject,
        policy_signature=policy_signature,
        coordinate_signature=baseline_coordinate_signature,
        operational_profile=operational_profile,
    )

    conversation_subject = initial_context_package_subject(
        user_persona_id=_optional_text(conversation.get("user_persona_id")),
        platform_id=_optional_text(conversation.get("platform_id")),
        character_id=_optional_text(conversation.get("character_id")),
        workspace_id=_optional_text(conversation.get("workspace_id")),
        assistant_mode_id=(
            _optional_text(conversation.get("assistant_mode_id"))
            or retrieval_profile_id
        ),
        mode=_optional_text(conversation.get("mode")),
    )
    conversation_coordinate_signature = (
        await build_initial_context_package_coordinate_signature(
            connection,
            user_id=user_id,
            retrieval_profile_id=retrieval_profile_id,
            conversation_id=conversation_id,
            conversation=conversation,
        )
    )
    conversation_key = build_initial_context_package_key(
        version=INITIAL_CONTEXT_PACKAGE_SCHEMA_VERSION,
        package_kind=InitialContextPackageKind.CONVERSATION,
        user_id=user_id,
        conversation_id=conversation_id,
        retrieval_profile_id=retrieval_profile_id,
        subject_json=conversation_subject,
        policy_signature=policy_signature,
        coordinate_signature=conversation_coordinate_signature,
        operational_profile=operational_profile,
    )
    return [
        _ExpectedPackage(
            kind=InitialContextPackageKind.BASELINE,
            user_id=user_id,
            key_hash=initial_context_package_key_hash(baseline_key),
            subject_json=baseline_subject,
            retrieval_profile_id=retrieval_profile_id,
            conversation_id=None,
        ),
        _ExpectedPackage(
            kind=InitialContextPackageKind.CONVERSATION,
            user_id=user_id,
            key_hash=initial_context_package_key_hash(conversation_key),
            subject_json=conversation_subject,
            retrieval_profile_id=retrieval_profile_id,
            conversation_id=conversation_id,
        ),
    ]


async def _read_expected_package(
    repository: InitialContextPackageRepository,
    request: _ExpectedPackage,
) -> _PackageRead:
    read_result = await repository.read_by_key_hash(
        user_id=request.user_id,
        package_key_hash=request.key_hash,
    )
    if read_result.status != "miss":
        package = read_result.package if read_result.status == "hit" else None
        status = _usable_status(read_result.status, read_result.package)
        if status != "hit":
            package = None
        return _PackageRead(
            package,
            _package_read_diagnostics(
                request,
                status=status,
                package=read_result.package,
                fallback_reason=read_result.fallback_reason,
            ),
        )

    latest = await _latest_related_package(repository, request)
    if latest is None:
        return _PackageRead(
            None,
            _package_read_diagnostics(request, status="miss", package=None),
        )
    if latest.build_status != InitialContextPackageBuildStatus.ACTIVE:
        return _PackageRead(
            None,
            _package_read_diagnostics(
                request,
                status=latest.build_status.value,
                package=latest,
                fallback_reason=f"package_{latest.build_status.value}",
            ),
        )
    return _PackageRead(
        None,
        _package_read_diagnostics(
            request,
            status="signature_mismatch",
            package=latest,
            fallback_reason="package_key_hash_mismatch",
        ),
    )


async def _latest_related_package(
    repository: InitialContextPackageRepository,
    request: _ExpectedPackage,
) -> InitialContextPackageRecord | None:
    if request.kind == InitialContextPackageKind.BASELINE:
        return await repository.get_latest_for_baseline_subject(
            user_id=request.user_id,
            retrieval_profile_id=request.retrieval_profile_id,
            subject_json=request.subject_json,
            include_inactive=True,
        )
    if request.conversation_id is None:
        return None
    return await repository.get_latest_for_conversation(
        user_id=request.user_id,
        conversation_id=request.conversation_id,
        retrieval_profile_id=request.retrieval_profile_id,
        include_inactive=True,
    )


def _usable_status(
    status: str,
    package: InitialContextPackageRecord | None,
) -> str:
    if status != "hit" or package is None:
        return status
    if package.version != INITIAL_CONTEXT_PACKAGE_SCHEMA_VERSION:
        return "version_mismatch"
    if not package.coordinate_signature_json.complete:
        return "coordinate_incomplete"
    return "hit"


def _package_read_diagnostics(
    request: _ExpectedPackage,
    *,
    status: str,
    package: InitialContextPackageRecord | None,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "package_kind": request.kind.value,
        "status": status,
        "expected_key_hash": request.key_hash,
        "package_id": None if package is None else package.id,
        "package_key_hash": None if package is None else package.package_key_hash,
        "package_version": None if package is None else package.version,
        "updated_at": None if package is None else package.updated_at,
        "fallback_reason": fallback_reason,
    }


def _render_packages(
    packages: list[InitialContextPackageRecord],
    *,
    selected_source_keys: set[tuple[str, str]],
    live_topic_present: bool,
    live_contract_present: bool,
    live_state_present: bool,
    include_recent_verbatim_seed: bool,
    budget_tokens: int,
) -> _RenderResult:
    lines = [
        "[Prepared Initial Context]",
        (
            "Use this as background orientation only. Query-specific retrieved "
            "evidence and the recent transcript outrank this block."
        ),
    ]
    remaining_budget = budget_tokens - estimate_tokens("\n".join(lines))
    diagnostics = {
        "rendered": False,
        "tokens_estimate": 0,
        "selected_profile_items": 0,
        "dropped_profile_items": 0,
        "selected_curated_items": 0,
        "dropped_curated_items": 0,
        "deduped_source_refs": 0,
        "dropped_sections": [],
        "known_empty": {},
    }
    if remaining_budget <= 0:
        diagnostics["dropped_sections"].append("package_header")
        return _RenderResult("", diagnostics)

    for package in packages:
        for key, value in package.blocks_json.empty_markers.items():
            diagnostics["known_empty"][key] = (
                bool(diagnostics["known_empty"].get(key)) or bool(value)
            )
        section = _render_package(
            package,
            selected_source_keys=selected_source_keys,
            live_topic_present=live_topic_present,
            live_contract_present=live_contract_present,
            live_state_present=live_state_present,
            include_recent_verbatim_seed=include_recent_verbatim_seed,
            remaining_budget=remaining_budget,
        )
        diagnostics["selected_profile_items"] += section["selected_profile_items"]
        diagnostics["dropped_profile_items"] += section["dropped_profile_items"]
        diagnostics["selected_curated_items"] += section["selected_curated_items"]
        diagnostics["dropped_curated_items"] += section["dropped_curated_items"]
        diagnostics["deduped_source_refs"] += section["deduped_source_refs"]
        diagnostics["dropped_sections"].extend(section["dropped_sections"])
        body = str(section["body"])
        if not body:
            continue
        body_tokens = estimate_tokens(body)
        if body_tokens > remaining_budget:
            diagnostics["dropped_sections"].append(
                f"{package.package_kind.value}:budget"
            )
            continue
        lines.append(body)
        remaining_budget -= body_tokens

    block = "\n\n".join(lines) if len(lines) > 2 else ""
    diagnostics["rendered"] = bool(block)
    diagnostics["tokens_estimate"] = estimate_tokens(block) if block else 0
    return _RenderResult(block, diagnostics)


def _render_package(
    package: InitialContextPackageRecord,
    *,
    selected_source_keys: set[tuple[str, str]],
    live_topic_present: bool,
    live_contract_present: bool,
    live_state_present: bool,
    include_recent_verbatim_seed: bool,
    remaining_budget: int,
) -> dict[str, Any]:
    blocks = package.blocks_json
    body_lines = [
        f"[{package.package_kind.value.title()} Prepared Context]",
        f"updated_at: {package.updated_at}",
    ]
    dropped_sections: list[str] = []
    deduped_source_refs = 0
    selected_profile_items = 0
    dropped_profile_items = 0
    selected_curated_items = 0
    dropped_curated_items = 0

    def add_section(name: str, text: str, *, source_refs: list[dict[str, Any]] | None = None) -> None:
        nonlocal deduped_source_refs
        normalized = text.strip()
        if not normalized:
            return
        refs = source_refs or []
        if refs and _all_refs_duplicate(refs, selected_source_keys):
            deduped_source_refs += len(refs)
            dropped_sections.append(f"{package.package_kind.value}:{name}:deduped")
            return
        body_lines.append(normalized)

    source_refs = package.source_refs_json
    if not live_contract_present:
        add_section(
            "contract",
            blocks.contract_block,
            source_refs=list(source_refs.get("contract") or []),
        )
    elif blocks.contract_block:
        dropped_sections.append(f"{package.package_kind.value}:contract:live_present")
    add_section("coordinates", blocks.coordinate_context_block)
    curated = _render_curated_items(
        blocks.curated_items,
        selected_source_keys=selected_source_keys,
        include_recent_verbatim_seed=include_recent_verbatim_seed,
        budget_tokens=max(0, remaining_budget - estimate_tokens("\n".join(body_lines))),
    )
    if curated["body"]:
        body_lines.append(str(curated["body"]))
    elif blocks.curated_orientation_block and not blocks.curated_items:
        curated_refs = list(source_refs.get("curated_orientation") or [])
        if (
            not include_recent_verbatim_seed
            and _has_message_ref(curated_refs)
        ):
            dropped_sections.append(
                f"{package.package_kind.value}:curated_orientation:recent_seed_disabled"
            )
        else:
            add_section(
                "curated_orientation",
                blocks.curated_orientation_block,
                source_refs=curated_refs,
            )
    selected_curated_items += int(curated["selected_curated_items"])
    dropped_curated_items += int(curated["dropped_curated_items"])
    deduped_source_refs += int(curated["deduped_source_refs"])
    add_section(
        "summary",
        blocks.conversation_summary_block,
        source_refs=list(source_refs.get("conversation_summary") or []),
    )
    if not live_topic_present:
        add_section(
            "topic",
            blocks.working_topic_block,
            source_refs=list(source_refs.get("working_topic") or []),
        )
    elif blocks.working_topic_block:
        dropped_sections.append(f"{package.package_kind.value}:topic:live_present")
    if not live_state_present:
        add_section(
            "state",
            blocks.current_state_block,
            source_refs=list(source_refs.get("current_state") or []),
        )
    elif blocks.current_state_block:
        dropped_sections.append(f"{package.package_kind.value}:state:live_present")

    profile = _render_profile_items(
        blocks.profile_items,
        selected_source_keys=selected_source_keys,
        budget_tokens=max(0, remaining_budget - estimate_tokens("\n".join(body_lines))),
    )
    if profile["body"]:
        body_lines.append(str(profile["body"]))
    selected_profile_items += int(profile["selected_profile_items"])
    dropped_profile_items += int(profile["dropped_profile_items"])
    deduped_source_refs += int(profile["deduped_source_refs"])

    if include_recent_verbatim_seed:
        recent_seed = _render_recent_seed(
            blocks.recent_verbatim_seed,
            selected_source_keys=selected_source_keys,
        )
        if recent_seed["body"]:
            body_lines.append(str(recent_seed["body"]))
        deduped_source_refs += int(recent_seed["deduped_source_refs"])
        if recent_seed["dropped"]:
            dropped_sections.append(f"{package.package_kind.value}:recent_seed:deduped")
    elif blocks.recent_verbatim_seed:
        dropped_sections.append(f"{package.package_kind.value}:recent_seed:disabled")

    return {
        "body": "\n".join(body_lines) if len(body_lines) > 2 else "",
        "selected_profile_items": selected_profile_items,
        "dropped_profile_items": dropped_profile_items,
        "selected_curated_items": selected_curated_items,
        "dropped_curated_items": dropped_curated_items,
        "deduped_source_refs": deduped_source_refs,
        "dropped_sections": dropped_sections,
    }


def _render_profile_items(
    items: list[InitialContextPackageProfileItem],
    *,
    selected_source_keys: set[tuple[str, str]],
    budget_tokens: int,
) -> dict[str, Any]:
    if not items or budget_tokens <= 0:
        return {
            "body": "",
            "selected_profile_items": 0,
            "dropped_profile_items": len(items),
            "deduped_source_refs": 0,
        }
    lines = [
        "[Prepared Memory Profile]",
        (
            "Stable orientation only; direct retrieved evidence wins on exact "
            "facts and conflicts."
        ),
    ]
    selected = 0
    dropped = 0
    deduped = 0
    for item in items:
        ref_keys = _source_ref_keys(item.source_refs)
        if ref_keys and ref_keys.issubset(selected_source_keys):
            deduped += len(ref_keys)
            dropped += 1
            continue
        line = _profile_item_line(item)
        candidate = "\n".join([*lines, line])
        if estimate_tokens(candidate) > budget_tokens:
            dropped += 1
            continue
        lines.append(line)
        selected += 1
    return {
        "body": "\n".join(lines) if selected else "",
        "selected_profile_items": selected,
        "dropped_profile_items": dropped,
        "deduped_source_refs": deduped,
    }


def _render_curated_items(
    items: list[InitialContextPackageProfileItem],
    *,
    selected_source_keys: set[tuple[str, str]],
    include_recent_verbatim_seed: bool,
    budget_tokens: int,
) -> dict[str, Any]:
    if not items or budget_tokens <= 0:
        return {
            "body": "",
            "selected_curated_items": 0,
            "dropped_curated_items": len(items),
            "deduped_source_refs": 0,
        }
    lines = [
        "[Curated Initial Orientation]",
        (
            "Source-grounded background orientation from package refresh; live "
            "evidence and recent transcript win on exact facts and conflicts."
        ),
    ]
    selected = 0
    dropped = 0
    deduped = 0
    for item in items:
        if (
            not include_recent_verbatim_seed
            and _has_message_ref(item.source_refs)
        ):
            dropped += 1
            continue
        ref_keys = _source_ref_keys(item.source_refs)
        if ref_keys and ref_keys.issubset(selected_source_keys):
            deduped += len(ref_keys)
            dropped += 1
            continue
        line = _profile_item_line(item)
        candidate = "\n".join([*lines, line])
        if estimate_tokens(candidate) > budget_tokens:
            dropped += 1
            continue
        lines.append(line)
        selected += 1
    return {
        "body": "\n".join(lines) if selected else "",
        "selected_curated_items": selected,
        "dropped_curated_items": dropped,
        "deduped_source_refs": deduped,
    }


def _profile_item_line(item: InitialContextPackageProfileItem) -> str:
    refs = ", ".join(
        str(ref.get("memory_id") or ref.get("message_id") or ref.get("source_kind"))
        for ref in item.source_refs[:2]
    )
    scope = item.scope_json.get("scope_canonical") or item.scope_json.get("scope")
    suffix_parts = [f"source: {refs}"] if refs else []
    if scope:
        suffix_parts.append(f"scope: {scope}")
    if item.status != "current":
        suffix_parts.append(f"status: {item.status}")
    suffix = f" ({'; '.join(suffix_parts)})" if suffix_parts else ""
    prefix = f"[{item.status}] " if item.status != "current" else ""
    return f"- {prefix}{item.text}{suffix}"


def _render_recent_seed(
    seeds: list[dict[str, Any]],
    *,
    selected_source_keys: set[tuple[str, str]],
) -> dict[str, Any]:
    if not seeds:
        return {"body": "", "deduped_source_refs": 0, "dropped": False}
    lines = ["[Prepared Recent Verbatim Seed]"]
    deduped = 0
    for seed in seeds:
        message_id = _optional_text(seed.get("message_id"))
        if message_id is not None and ("message", message_id) in selected_source_keys:
            deduped += 1
            continue
        text = _optional_text(seed.get("text"))
        if text is None:
            continue
        role = _optional_text(seed.get("role")) or "user"
        seq = seed.get("seq")
        lines.append(f"- seq {seq} {role}: {text}")
    return {
        "body": "\n".join(lines) if len(lines) > 1 else "",
        "deduped_source_refs": deduped,
        "dropped": deduped > 0,
    }


def _available_package_budget(
    context_envelope_budget: ContextEnvelopeBudget,
    *,
    retrieved_context_tokens: int,
    topic_context_block: str,
    prompt_budget_tokens: int,
) -> int:
    topic_tokens = estimate_tokens(topic_context_block)
    retrieved_remaining = (
        context_envelope_budget.retrieved_context_budget_tokens
        - int(retrieved_context_tokens)
        - topic_tokens
    )
    return max(0, min(int(prompt_budget_tokens), retrieved_remaining))


def _baseline_subject(
    conversation: Mapping[str, Any],
    retrieval_profile_id: str,
) -> dict[str, str | None]:
    workspace_id = _optional_text(conversation.get("workspace_id"))
    return initial_context_package_subject(
        user_persona_id=_optional_text(conversation.get("user_persona_id")),
        platform_id=_optional_text(conversation.get("platform_id")),
        character_id=_optional_text(conversation.get("character_id")) or workspace_id,
        workspace_id=workspace_id,
        assistant_mode_id=retrieval_profile_id,
        mode=_optional_text(conversation.get("mode")) or retrieval_profile_id,
    )


def _authority_requires_live_context(authority_context: PromptAuthorityContext) -> bool:
    return (
        authority_context.normalized_privilege_level != "standard"
        or authority_context.authenticated_user_is_atagia_master
        or authority_context.trusted_evaluation
    )


def _source_ref_keys(refs: list[dict[str, Any]]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for ref in refs:
        memory_id = _optional_text(ref.get("memory_id"))
        if memory_id is not None:
            keys.add(("memory", memory_id))
        message_id = _optional_text(ref.get("message_id"))
        if message_id is not None:
            keys.add(("message", message_id))
        summary_id = _optional_text(ref.get("summary_id"))
        if summary_id is not None:
            keys.add(("summary", summary_id))
        topic_id = _optional_text(ref.get("topic_id"))
        if topic_id is not None:
            keys.add(("topic", topic_id))
    return keys


def _all_refs_duplicate(
    refs: list[dict[str, Any]],
    selected_source_keys: set[tuple[str, str]],
) -> bool:
    keys = _source_ref_keys(refs)
    return bool(keys) and keys.issubset(selected_source_keys)


def _has_message_ref(refs: list[dict[str, Any]]) -> bool:
    keys = _source_ref_keys(refs)
    return any(kind == "message" for kind, _ in keys)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _coerce_bool(value: Any) -> bool:
    return bool(value)


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000.0, 3)
