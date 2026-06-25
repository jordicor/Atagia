"""Parallel-card need detection for retrieval planning."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import html
from typing import Any, Literal

from atagia.core.clock import Clock
from atagia.core.config import Settings
from atagia.memory.card_prompt import compose_card_prompt
from atagia.memory.policy_manifest import ResolvedRetrievalPolicy
from atagia.models.schemas_memory import (
    DetectedNeed,
    ExactFacet,
    ExtractionConversationContext,
    MemoryDependence,
    NeedTrigger,
    QueryIntelligenceResult,
    RuntimeAnchor,
    RuntimeAnchorAlias,
    UserCommunicationProfile,
)
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMMessage,
    known_intimacy_context_metadata,
)
from atagia.services.model_resolution import (
    examples_enabled_for_component,
    resolve_component_model,
)
from atagia.services.prompt_authority import (
    PromptAuthorityContext,
    process_authority_context,
    prompt_authority_metadata,
    render_process_metadata_block,
)

NeedCardName = Literal[
    "needs",
    "language",
    "memory",
    "exact",
    "shape",
    "facets",
    "callback",
    "search_words",
    "search_words_other_language",
]

_CARD_NAMES: tuple[NeedCardName, ...] = (
    "needs",
    "language",
    "memory",
    "exact",
    "shape",
    "facets",
    "callback",
    "search_words",
)

_CARD_COMPONENT_IDS: dict[NeedCardName, str] = {
    "needs": "need_detector_needs",
    "language": "need_detector_language",
    "memory": "need_detector_memory",
    "exact": "need_detector_exact",
    "shape": "need_detector_shape",
    "facets": "need_detector_facets",
    "callback": "need_detector_callback",
    "search_words": "need_detector_search_words",
    "search_words_other_language": "need_detector_search_words_other_language",
}

_CARD_PURPOSES: dict[NeedCardName, str] = {
    "needs": "need_detection_needs_card",
    "language": "need_detection_language_card",
    "memory": "need_detection_memory_card",
    "exact": "need_detection_exact_card",
    "shape": "need_detection_shape_card",
    "facets": "need_detection_facets_card",
    "callback": "need_detection_callback_card",
    "search_words": "need_detection_search_words_card",
    "search_words_other_language": "need_detection_search_words_other_language_card",
}

_NEED_DESCRIPTIONS: dict[NeedTrigger, str] = {
    NeedTrigger.AMBIGUITY: "The user request is unclear, underspecified, or could be interpreted in multiple ways.",
    NeedTrigger.CONTRADICTION: "The current request conflicts with prior stated preferences, facts, or actions.",
    NeedTrigger.FOLLOW_UP_FAILURE: "A follow-up suggests prior advice or action did not solve the problem.",
    NeedTrigger.LOOP: "The conversation is circling the same unresolved issue or blocker.",
    NeedTrigger.HIGH_STAKES: "The request has meaningful legal, medical, safety, financial, or other serious consequences.",
    NeedTrigger.MODE_SHIFT: "The user appears to be switching task mode or desired interaction style.",
    NeedTrigger.FRUSTRATION: "The user sounds frustrated, impatient, or destabilized by the interaction.",
    NeedTrigger.SENSITIVE_CONTEXT: "The message touches on privacy-sensitive or emotionally delicate context.",
    NeedTrigger.UNDER_SPECIFIED_REQUEST: "The user asks for help without enough constraints, goals, or success criteria.",
}

_FACET_MAP: dict[str, ExactFacet] = {
    "date": ExactFacet.DATE,
    "time": ExactFacet.DATE,
    "phone": ExactFacet.PHONE,
    "email": ExactFacet.EMAIL,
    "code": ExactFacet.CODE,
    "location": ExactFacet.LOCATION,
    "address": ExactFacet.LOCATION,
    "quantity": ExactFacet.QUANTITY,
    "number": ExactFacet.QUANTITY,
    "person": ExactFacet.PERSON_NAME,
    "person_name": ExactFacet.PERSON_NAME,
    "organization": ExactFacet.ORG_NAME,
    "org": ExactFacet.ORG_NAME,
    "org_name": ExactFacet.ORG_NAME,
    "medication": ExactFacet.MEDICATION,
    "medicine": ExactFacet.MEDICATION,
    "wording": ExactFacet.OTHER_VERBATIM,
    "other_verbatim": ExactFacet.OTHER_VERBATIM,
}

_ALIAS_KIND_VALUES = {
    "translation",
    "transliteration",
    "spelling_variant",
    "acronym_expansion",
    "domain_synonym",
    "corpus_surface",
}


@dataclass(slots=True)
class NeedCardCall:
    card_name: NeedCardName
    model: str
    prompt: str | None = None
    raw_output: str | None = None
    parsed: dict[str, Any] = field(default_factory=dict)
    parse_valid: bool = False
    error: str | None = None


class NeedDetector:
    """LLM-backed detector that decomposes query understanding into cards."""

    def __init__(
        self,
        llm_client: LLMClient[Any],
        clock: Clock,
        settings: Settings | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._clock = clock
        resolved_settings = settings or Settings.from_env()
        self._card_models = {
            card_name: resolve_component_model(
                resolved_settings,
                component_id,
            )
            for card_name, component_id in _CARD_COMPONENT_IDS.items()
        }
        self._card_include_examples = {
            card_name: examples_enabled_for_component(resolved_settings, component_id)
            for card_name, component_id in _CARD_COMPONENT_IDS.items()
        }

    async def detect(
        self,
        message_text: str,
        role: str,
        conversation_context: ExtractionConversationContext | dict[str, Any],
        resolved_policy: ResolvedRetrievalPolicy,
        content_language_profile: list[dict[str, Any]],
        user_communication_profile: UserCommunicationProfile | None = None,
        prompt_authority_context: PromptAuthorityContext | None = None,
        card_call_trace_sink: list[NeedCardCall] | None = None,
    ) -> QueryIntelligenceResult:
        if content_language_profile is None:
            raise ValueError(
                "content_language_profile must be provided; pass an empty list when unknown"
            )
        context = ExtractionConversationContext.model_validate(conversation_context)
        authority_context = (
            prompt_authority_context
            or _authority_context_from_extraction_context(
                context,
                purpose="need_detection",
            )
        )
        base_calls = await asyncio.gather(
            *(
                self._run_card(
                    card_name=card_name,
                    message_text=message_text,
                    role=role,
                    context=context,
                    resolved_policy=resolved_policy,
                    content_language_profile=content_language_profile,
                    user_communication_profile=user_communication_profile,
                    prompt_authority_context=authority_context,
                )
                for card_name in _CARD_NAMES
            )
        )
        calls = list(base_calls)
        if _should_run_search_words_other_language(calls, content_language_profile):
            search_words = _normalize_text_list(
                _calls_by_card(calls)
                .get("search_words", NeedCardCall("search_words", ""))
                .parsed.get("anchor_terms")
                or []
            )
            calls.append(
                await self._run_card(
                    card_name="search_words_other_language",
                    message_text=message_text,
                    role=role,
                    context=context,
                    resolved_policy=resolved_policy,
                    content_language_profile=content_language_profile,
                    user_communication_profile=user_communication_profile,
                    prompt_authority_context=authority_context,
                    search_words=search_words,
                )
            )
        if card_call_trace_sink is not None:
            card_call_trace_sink.extend(calls)
        if all(not call.parse_valid for call in calls):
            errors = "; ".join(
                call.error or f"{call.card_name}: invalid output"
                for call in calls
            )
            raise ValueError(f"all need detection cards failed: {errors}")
        result = self._merge_cards(
            message_text=message_text,
            context=context,
            resolved_policy=resolved_policy,
            content_language_profile=content_language_profile,
            calls=list(calls),
        )
        return result

    async def _run_card(
        self,
        *,
        card_name: NeedCardName,
        message_text: str,
        role: str,
        context: ExtractionConversationContext,
        resolved_policy: ResolvedRetrievalPolicy,
        content_language_profile: list[dict[str, Any]],
        user_communication_profile: UserCommunicationProfile | None,
        prompt_authority_context: PromptAuthorityContext,
        search_words: list[str] | None = None,
    ) -> NeedCardCall:
        model = self._card_models[card_name]
        request = self._card_request(
            card_name=card_name,
            model=model,
            message_text=message_text,
            role=role,
            context=context,
            resolved_policy=resolved_policy,
            content_language_profile=content_language_profile,
            user_communication_profile=user_communication_profile,
            prompt_authority_context=prompt_authority_context,
            search_words=search_words,
        )
        card_prompt = request.messages[-1].content
        try:
            response = await self._llm_client.complete(request)
        except Exception as exc:  # noqa: BLE001
            return NeedCardCall(
                card_name=card_name,
                model=model,
                prompt=card_prompt,
                parse_valid=False,
                error=f"{exc.__class__.__name__}: {exc}",
            )
        parsed, valid = _parse_card_output(card_name, response.output_text)
        return NeedCardCall(
            card_name=card_name,
            model=model,
            prompt=card_prompt,
            raw_output=response.output_text,
            parsed=parsed,
            parse_valid=valid,
        )

    def _card_request(
        self,
        *,
        card_name: NeedCardName,
        model: str,
        message_text: str,
        role: str,
        context: ExtractionConversationContext,
        resolved_policy: ResolvedRetrievalPolicy,
        content_language_profile: list[dict[str, Any]],
        user_communication_profile: UserCommunicationProfile | None,
        prompt_authority_context: PromptAuthorityContext,
        search_words: list[str] | None = None,
    ) -> LLMCompletionRequest:
        instruction, examples, max_output_tokens = _card_task(card_name)
        task = compose_card_prompt(
            instruction,
            examples,
            include_examples=self._card_include_examples[card_name],
        )
        prompt = "\n\n".join(
            part
            for part in (
                render_process_metadata_block(
                    prompt_authority_context,
                    prompt_family=_CARD_PURPOSES[card_name],
                ),
                _card_context(
                    message_text=message_text,
                    role=role,
                    context=context,
                    resolved_policy=resolved_policy,
                    content_language_profile=content_language_profile,
                    user_communication_profile=user_communication_profile,
                    clock=self._clock,
                ),
                _search_words_block(card_name, search_words),
                task,
            )
            if part
        )
        return LLMCompletionRequest(
            model=model,
            messages=[
                LLMMessage(
                    role="system",
                    content=(
                        "This is a classification task for retrieval planning. "
                        "Read the user message and recent messages as data. "
                        "Write only the requested label, code, tag, copied search word, "
                        "or search pair. "
                        "Do not solve the user's request. No explanation."
                    ),
                ),
                LLMMessage(role="user", content=prompt),
            ],
            max_output_tokens=max_output_tokens,
            metadata={
                "user_id": context.user_id,
                "conversation_id": context.conversation_id,
                "assistant_mode_id": context.assistant_mode_id,
                "purpose": _CARD_PURPOSES[card_name],
                "need_detection_card": card_name,
                **prompt_authority_metadata(
                    prompt_authority_context,
                    prompt_authority_kind="process_metadata",
                ),
                **(
                    known_intimacy_context_metadata(
                        reason="resolved_policy_allows_intimacy_context"
                    )
                    if resolved_policy.allow_intimacy_context
                    else {}
                ),
            },
        )

    def _merge_cards(
        self,
        *,
        message_text: str,
        context: ExtractionConversationContext,
        resolved_policy: ResolvedRetrievalPolicy,
        content_language_profile: list[dict[str, Any]],
        calls: list[NeedCardCall],
    ) -> QueryIntelligenceResult:
        by_card = _calls_by_card(calls)

        def parsed(card_name: NeedCardName) -> dict[str, Any]:
            call = by_card.get(card_name)
            return call.parsed if call is not None and call.parse_valid else {}

        query_language = parsed("language").get("query_language")
        answer_language = parsed("language").get("answer_language")
        memory_dependence = parsed("memory").get("memory_dependence")
        if memory_dependence not in {item.value for item in MemoryDependence}:
            memory_dependence = MemoryDependence.MIXED.value

        exact_recall_needed = parsed("exact").get("exact_recall_needed")
        if exact_recall_needed is None:
            exact_recall_needed = memory_dependence in {
                MemoryDependence.PERSONAL.value,
                MemoryDependence.MIXED.value,
            }

        if memory_dependence == MemoryDependence.CONVERSATION.value and not context.recent_messages:
            memory_dependence = (
                MemoryDependence.PERSONAL.value
                if exact_recall_needed
                else MemoryDependence.MIXED.value
            )
        # A skip-class label (world/conversation) is contradicted only when the
        # turn needs a specific SAVED detail: exact recall AND a saved-detail
        # shape (slot/list/time). This keeps pure conversation turns (e.g.
        # "summarize what you just said", shape=default) skippable while
        # rescuing stored-attribute questions the card mislabeled as world or
        # conversation.
        shape_needs_saved_detail = parsed("shape").get("query_type") in {
            "slot_fill",
            "broad_list",
            "temporal",
        }
        if (
            exact_recall_needed
            and shape_needs_saved_detail
            and memory_dependence in {
                MemoryDependence.WORLD.value,
                MemoryDependence.CONVERSATION.value,
            }
        ):
            memory_dependence = MemoryDependence.MIXED.value

        query_type = parsed("shape").get("query_type") or "default"
        if memory_dependence in {
            MemoryDependence.WORLD.value,
            MemoryDependence.CONVERSATION.value,
        } and not exact_recall_needed:
            query_type = "default"
        # Keep this default->slot_fill promotion BELOW the shape-gated override
        # above: that override reads the RAW shape value, so promoting here first
        # would make it fire on every exact-recall turn.
        if query_type == "default" and exact_recall_needed:
            query_type = "slot_fill"

        exact_facets = _normalize_exact_facets(parsed("facets").get("exact_facets") or [])
        if exact_facets:
            exact_recall_needed = True
        if not exact_recall_needed:
            exact_facets = []

        callback_bias = parsed("callback").get("callback_bias")
        if callback_bias is None:
            callback_bias = False

        anchor_card = parsed("search_words")
        anchor_terms = _normalize_text_list(anchor_card.get("anchor_terms") or [])
        target_content_languages = _content_language_codes(content_language_profile)
        anchor_aliases = _anchor_aliases_by_search_word(
            parsed("search_words_other_language").get("anchor_alias_pairs") or [],
            anchor_terms=anchor_terms,
            query_language=query_language,
            target_content_languages=target_content_languages,
        )
        original_query = message_text.strip() or "(empty query)"
        must_keep_terms = list(anchor_terms)
        if (callback_bias or query_type in {"slot_fill", "broad_list"} or exact_recall_needed) and not must_keep_terms:
            must_keep_terms = [original_query]

        raw_context_access_mode = (
            "verbatim"
            if callback_bias or ExactFacet.OTHER_VERBATIM in exact_facets
            else "normal"
        )
        needs = _filter_detected_needs(
            parsed("needs").get("needs") or [],
            resolved_policy,
        )
        anchors = [
            RuntimeAnchor(
                sub_query_text=original_query,
                anchor_type="unknown",
                original_surface=term,
                preserve_verbatim=True,
                aliases=_runtime_anchor_aliases(
                    anchor_aliases.get(term) or [],
                    target_content_languages=target_content_languages,
                    source="need_detection_search_words_other_language_card",
                ),
                confidence=0.5,
                derivation={"source": "need_detection_search_words_card"},
            )
            for term in anchor_terms
        ]

        return QueryIntelligenceResult(
            needs=needs,
            temporal_range=None,
            sub_queries=[original_query],
            callback_bias=bool(callback_bias),
            raw_context_access_mode=raw_context_access_mode,
            sparse_query_hints=[
                {
                    "sub_query_text": original_query,
                    "fts_phrase": original_query,
                    "must_keep_terms": must_keep_terms,
                }
            ],
            query_language=query_language,
            answer_language=answer_language,
            anchors=anchors,
            query_type=query_type,
            memory_dependence=MemoryDependence(memory_dependence),
            retrieval_levels=[0],
            exact_recall_needed=bool(exact_recall_needed),
            exact_facets=exact_facets,
        )

    @staticmethod
    def _summarize_content_language_profile(
        content_language_profile: list[dict[str, Any]],
    ) -> str:
        if not content_language_profile:
            return "(none)"
        lines: list[str] = []
        for row in content_language_profile:
            language_code = (
                str(row.get("language_code", "")).strip().lower() or "unknown"
            )
            memory_count = int(row.get("memory_count", 0))
            raw_last_seen_at = str(row.get("last_seen_at", "")).strip()
            last_seen_date = (
                raw_last_seen_at[:10]
                if len(raw_last_seen_at) >= 10
                else raw_last_seen_at or "unknown"
            )
            line = (
                f"{language_code}: {memory_count} memories (last seen {last_seen_date})"
            )
            if any(character in raw_last_seen_at for character in "<>&"):
                line = f"{line} raw={raw_last_seen_at}"
            lines.append(html.escape(line))
        return "\n".join(lines)

    @staticmethod
    def _summarize_user_communication_profile(
        user_communication_profile: UserCommunicationProfile | None,
    ) -> str:
        if user_communication_profile is None or user_communication_profile.stale:
            return "(none)"
        lines: list[str] = [
            "control_plane_only=true",
            "not_factual_answer_evidence=true",
        ]
        if user_communication_profile.observed_user_languages:
            observed = ", ".join(
                html.escape(
                    (
                        f"{row.language_code}: {row.message_count} observed "
                        "user-authored messages"
                    )
                )
                for row in user_communication_profile.observed_user_languages
            )
            lines.append(f"observed_user_languages: {observed}")
        if user_communication_profile.explicit_language_preferences:
            preferences = ", ".join(
                html.escape(
                    (
                        f"{row.language_code}/{row.preference_kind}"
                        f"/{row.context_label}"
                    )
                )
                for row in user_communication_profile.explicit_language_preferences
            )
            lines.append(f"explicit_language_preferences: {preferences}")
        if user_communication_profile.explicit_language_abilities:
            abilities = ", ".join(
                html.escape(f"{row.language_code}/{row.ability_kind}")
                for row in user_communication_profile.explicit_language_abilities
            )
            lines.append(f"explicit_language_abilities: {abilities}")
        if user_communication_profile.contextual_norms:
            norms = ", ".join(
                html.escape(
                    f"{row.language_code}/{row.norm_kind}/{row.context_label}"
                )
                for row in user_communication_profile.contextual_norms
            )
            lines.append(f"contextual_norms: {norms}")
        return "\n".join(lines)


def _card_context(
    *,
    message_text: str,
    role: str,
    context: ExtractionConversationContext,
    resolved_policy: ResolvedRetrievalPolicy,
    content_language_profile: list[dict[str, Any]],
    user_communication_profile: UserCommunicationProfile | None,
    clock: Clock,
) -> str:
    recent_lines = [
        f"{message.role}: {message.content}"
        for message in context.recent_messages
    ]
    if not recent_lines:
        recent_lines = ["(none)"]
    if resolved_policy.need_triggers:
        need_types = ", ".join(need.value for need in resolved_policy.need_triggers)
        need_descriptions = "\n".join(
            f"- {need.value}: {_NEED_DESCRIPTIONS[need]}"
            for need in resolved_policy.need_triggers
        )
    else:
        need_types = "(none)"
        need_descriptions = "(none)"
    return "\n".join(
        [
            f"Reference time: {clock.now().isoformat()}",
            f"Role: {role}",
            f"User message: {message_text}",
            "Recent messages:",
            *recent_lines,
            "Saved memory languages:",
            NeedDetector._summarize_content_language_profile(content_language_profile),
            "User communication profile:",
            NeedDetector._summarize_user_communication_profile(user_communication_profile),
            f"Enabled need types: {need_types}",
            "Need type meanings:",
            need_descriptions,
        ]
    )


def _search_words_block(card_name: NeedCardName, search_words: list[str] | None) -> str:
    if card_name != "search_words_other_language":
        return ""
    words = _normalize_text_list(search_words or [])
    return "\n".join(
        [
            "Search words already chosen:",
            *(words if words else ["(none)"]),
        ]
    )


def _card_task(card_name: NeedCardName) -> tuple[str, str | None, int]:
    """Return (instruction, examples, max_output_tokens) for a card.

    The examples string holds only the demonstration pairs (no header); it is
    gated by the per-component examples toggle and joined by compose_card_prompt.
    """
    if card_name == "needs":
        return (
            "Read the user message and recent messages. Which enabled need types apply? "
            "Pick all that clearly fit, or none if no enabled need type fits.\n\n"
            "Use only names from Enabled need types. If Enabled need types is none, write: none\n"
            "Put one need type per line. No explanation. Do not invent need types.",
            "User message: I am stuck and the same fix failed again.\n"
            "follow_up_failure\n\n"
            "User message: I need a safe rollback plan for production data.\n"
            "high_stakes\n\n"
            "User message: Thanks, that worked.\n"
            "none",
            64,
        )
    if card_name == "language":
        return (
            "Read the user message. Write two language codes.\n\n"
            "Line 1 = the language of the user message.\n"
            "Line 2 = the language to answer in.\n\n"
            "Normally line 1 and line 2 are the same language.\n"
            "If the user asks to translate into a language, or asks to answer in a different language, "
            "then line 2 is that language. If the user profile contains an explicit answer-language preference "
            "that clearly applies, use that for line 2.\n\n"
            "Use two-letter ISO 639-1 codes. Examples: en, es, fr, de, it, pt, ru, ja, zh.\n\n"
            "Output only the two codes. One code per line. No spaces. No extra words.",
            "User message: Hola, que tal?\n"
            "es\n"
            "es\n\n"
            "User message: What time is it?\n"
            "en\n"
            "en\n\n"
            "User message: Translate \"house\" to French.\n"
            "en\n"
            "fr\n\n"
            "User message: Answer in English: donde estas?\n"
            "es\n"
            "en",
            24,
        )
    if card_name == "memory":
        return (
            "Read the user message. To answer it, what does it truly need? Pick one.\n\n"
            "personal = a fact the user told you before, saved from earlier chats: "
            "saved preferences, plans, private details, passwords, PIN codes, addresses, "
            "medicines, or past advice you gave them. Anything tied to the user's own "
            "life, history, or saved facts is personal, even when the topic also appears "
            "in the recent messages.\n"
            "conversation = answerable ENTIRELY from the recent messages alone, needing NO "
            "saved detail, for example summarizing or rephrasing what was just said.\n"
            "world = public, general knowledge with no tie to this user. Anyone could answer it.\n"
            "mixed = both apply, or it is unclear.\n\n"
            "Rules:\n"
            "- If the recent messages mention the topic but do not by themselves hold the "
            "final answer, choose personal or mixed, never conversation.\n"
            "- A personal anchor overrides the topic: a general-knowledge question tied to "
            "the user's own past is personal or mixed, never world.\n"
            "- When in doubt, choose mixed.\n\n"
            "Output only one word: personal, conversation, world, or mixed. Nothing else.",
            "How tall is Mount Everest? -> world\n"
            "Tell me a joke. -> world\n"
            "What did I just say? -> conversation\n"
            "Summarize the message I sent above. -> conversation\n"
            "Remind me what I told you to call me. -> personal\n"
            "We were discussing my schedule; when is my next appointment? -> personal\n"
            "What did we decide last week? -> personal\n"
            "We talked about my trip. What should I pack? -> mixed",
            20,
        )
    if card_name == "exact":
        return (
            "Read the user message. To answer it, do you need a specific fact or detail "
            "that the user told you before, in saved memory or in this chat?\n\n"
            "Say yes if the answer needs a remembered detail like a date, a dose, an address, "
            "a password, a PIN code, a preference, exact wording, a list the user made, "
            "or past advice you gave them.\n"
            "Say no if public, general knowledge is enough to answer.\n\n"
            "Output only one word: yes or no. Nothing else.",
            "What is my doctor's appointment date? -> yes\n"
            "How many days are in a week? -> no\n"
            "What dose of my pill do I take? -> yes\n"
            "What is the boiling point of water? -> no\n"
            "What was on my shopping list? -> yes\n"
            "Who wrote Romeo and Juliet? -> no\n"
            "Remind me what I told you to call me. -> yes",
            12,
        )
    if card_name == "shape":
        return (
            "Read the user message. What kind of saved detail does the answer need? Pick the best one.\n\n"
            "slot = one single exact value, like one address, one name, one password.\n"
            "list = several saved items together, like a group, a set, many things.\n"
            "time = a date, a time, or an order/sequence.\n"
            "default = no saved detail needed. Public knowledge, or just this chat, is enough.\n\n"
            "Output only one word: slot, list, time, or default. Nothing else.",
            "What is my wifi password? -> slot\n"
            "What are all the items on my list? -> list\n"
            "When is my dentist appointment? -> time\n"
            "In what order do I take my pills? -> time\n"
            "What is the capital of Spain? -> default\n"
            "What did I just say to you? -> default",
            16,
        )
    if card_name == "facets":
        return (
            "Read the user message. Which kinds of exact saved detail does the answer need? "
            "Pick all that fit, or none if no exact detail is needed.\n\n"
            "date = a date, day, or time.\n"
            "phone = a phone number.\n"
            "email = an email address.\n"
            "quantity = a number or amount, like a dose, count, price, or size.\n"
            "location = a place or address.\n"
            "person = a person's name.\n"
            "organization = an organization, company, team, or group name.\n"
            "medication = the name of a medicine or drug.\n"
            "code = technical text strings: API keys, prefixes, config strings, or software library names.\n"
            "wording = exact words the user wants kept, like a quote, slogan, or name.\n\n"
            "Output only the matching tags. Put one tag per line. If nothing fits, write: none",
            "What dose of aspirin do I take?\n"
            "quantity\n"
            "medication\n\n"
            "What is my home address?\n"
            "location\n\n"
            "When is my flight and what is my seat?\n"
            "date\n"
            "quantity\n\n"
            "What API key did I save?\n"
            "code\n\n"
            "Repeat the exact slogan I wrote.\n"
            "wording\n\n"
            "Tell me a joke.\n"
            "none",
            64,
        )
    if card_name == "callback":
        return (
            "Read the user message. Is the user asking about something you, the assistant, "
            "said or suggested earlier?\n\n"
            "Only count your own past answers and recommendations. Do not count what other people said or planned.\n"
            "Say yes if the user refers back to your earlier reply, like asking about advice, "
            "an idea, an answer, or a suggestion you gave them before.\n"
            "Say no if the user is asking something new, or asks about what another person said or planned.\n\n"
            "Output only one word: yes or no. Nothing else.",
            "What did you suggest I cook earlier? -> yes\n"
            "Tell me about the French Revolution. -> no\n"
            "Wait, go back to your last answer. -> yes\n"
            "What is the capital of Japan? -> no\n"
            "My boss said to finish by Friday. Is that ok? -> no\n"
            "Explain the plan you gave me again. -> yes",
            12,
        )
    if card_name == "search_words":
        return (
            "Read the user message and recent messages. Copy every useful word or short phrase to search for in saved notes later. "
            "Keep each word exactly as the user wrote it.\n\n"
            "Include words like these:\n"
            "- names of people, places, brands, products, medicines\n"
            "- codes, dates, numbers, and any phrase the user put in quotes\n"
            "- the main thing being asked about, such as an event, appointment, address, dose, setting, problem, project, or item\n\n"
            "Do not translate or change the words. Do not write an answer. Only copy words that already appear in "
            "the user message or recent messages.\n\n"
            "Write one word or short phrase per line, up to 6. If there are no useful search words, write: none",
            "User message: What is John's phone number?\n"
            "John\n\n"
            "User message: Did Dr. Morgan say to take aspirin on Friday?\n"
            "Dr. Morgan\n"
            "aspirin\n"
            "Friday\n\n"
            "User message: When is my flight to Tokyo?\n"
            "Tokyo\n\n"
            "User message: Remind me what \"Project Falcon\" was about.\n"
            "Project Falcon\n\n"
            "User message: Tell me a joke.\n"
            "none",
            64,
        )
    return (
        "Saved notes may use a different language than the user message. For each search word listed above, "
        "write the version that saved notes would likely use, if it has a normal other-language version.\n\n"
        "Only use words from the Search words list as the left side. Do not add new search words.\n"
        "Use pairs like: original word => other-language word\n\n"
        "Skip names, exact numbers, dates, addresses, IDs, passwords, and codes unless there is a normal spelling "
        "variant. If no word needs another version, write: none",
        "User message: What is John's phone number?\n"
        "Search words already chosen:\n"
        "John\n"
        "none\n\n"
        "User message: Cual es mi dosis de ibuprofeno?\n"
        "Search words already chosen:\n"
        "ibuprofeno\n"
        "ibuprofeno => ibuprofen\n\n"
        "User message: Cuando veo a mi medico de cabecera?\n"
        "Search words already chosen:\n"
        "medico de cabecera\n"
        "medico de cabecera => family doctor\n\n"
        "User message: What is code AB-190?\n"
        "Search words already chosen:\n"
        "AB-190\n"
        "none",
        64,
    )


def _parse_card_output(card_name: NeedCardName, text: str) -> tuple[dict[str, Any], bool]:
    stripped = (
        text.strip()
        .replace("<TAB>", "\t")
        .replace("<tab>", "\t")
        .replace("\\t", "\t")
    )
    if card_name == "language":
        pieces = _language_codes_from_output(stripped)
        query_language = pieces[0] if len(pieces) >= 1 else ""
        answer_language = pieces[1] if len(pieces) >= 2 else ""
        return {
            "query_language": query_language or None,
            "answer_language": answer_language or None,
        }, len(query_language) == 2 and len(answer_language) == 2
    if card_name == "needs":
        return _parse_needs(stripped)
    if card_name == "memory":
        value = _first_allowed_atom(
            stripped,
            {"personal", "conversation", "world", "mixed", "unclear", "public"},
        )
        if value == "unclear":
            value = "mixed"
        if value == "public":
            value = "world"
        return {"memory_dependence": value if value else None}, value in {
            "personal",
            "conversation",
            "world",
            "mixed",
        }
    if card_name == "exact":
        value, valid = _parse_yes_no(stripped)
        return {"exact_recall_needed": value}, valid
    if card_name == "shape":
        mapping = {
            "slot": "slot_fill",
            "list": "broad_list",
            "time": "temporal",
            "default": "default",
        }
        value = _first_allowed_atom(stripped, set(mapping))
        return {"query_type": mapping.get(value)}, value in mapping
    if card_name == "facets":
        return _parse_facets(stripped)
    if card_name == "callback":
        value, valid = _parse_yes_no(stripped)
        return {"callback_bias": value}, valid
    if card_name == "search_words":
        return _parse_search_words(stripped)
    return _parse_search_word_alias_pairs(stripped)


def _parse_facets(text: str) -> tuple[dict[str, Any], bool]:
    separators_normalized = text
    for separator in ("\n", ",", ";", "|"):
        separators_normalized = separators_normalized.replace(separator, "\t")
    raw_atoms = [
        _clean_atom(atom)
        for piece in separators_normalized.split("\t")
        for atom in piece.split()
    ]
    atoms = [atom for atom in raw_atoms if atom]
    if not atoms:
        return {"exact_facets": []}, False
    if atoms == ["none"] or "none" in atoms or atoms == ["no"]:
        return {"exact_facets": []}, True
    facets = _normalize_exact_facets(
        _FACET_MAP[atom]
        for raw_atom in atoms
        for atom in (raw_atom, raw_atom.split(":", 1)[0])
        if atom in _FACET_MAP
    )
    return {"exact_facets": facets}, bool(facets)


def _parse_needs(text: str) -> tuple[dict[str, Any], bool]:
    separators_normalized = text
    for separator in ("\n", ",", ";", "|"):
        separators_normalized = separators_normalized.replace(separator, "\t")
    raw_atoms = [
        _clean_atom(atom)
        for piece in separators_normalized.split("\t")
        for atom in piece.split()
    ]
    atoms = [atom for atom in raw_atoms if atom]
    if not atoms:
        return {"needs": []}, False
    if atoms == ["none"] or "none" in atoms or atoms == ["no"]:
        return {"needs": []}, True
    needs: list[DetectedNeed] = []
    seen: set[NeedTrigger] = set()
    allowed = {need.value: need for need in NeedTrigger}
    for atom in atoms:
        need = allowed.get(atom)
        if need is None or need in seen:
            continue
        seen.add(need)
        needs.append(
            DetectedNeed(
                need_type=need,
                confidence=0.7,
                reasoning="parallel need card",
            )
        )
    return {"needs": needs}, bool(needs)


def _parse_search_words(text: str) -> tuple[dict[str, Any], bool]:
    terms: list[str] = []
    source_lines = text.splitlines()
    saw_none = False
    if len(source_lines) == 1 and "," in text and "\t" not in text:
        source_lines = text.split(",")
    for line in source_lines:
        cleaned = line.strip().strip("-* ").strip()
        if not cleaned:
            continue
        if _clean_atom(cleaned) == "none":
            saw_none = True
            continue
        if "\t" in cleaned or "=>" in cleaned or "->" in cleaned:
            continue
        terms.append(cleaned)

    normalized_terms = _normalize_text_list(terms[:6])
    return {"anchor_terms": normalized_terms}, bool(normalized_terms) or saw_none


def _parse_search_word_alias_pairs(text: str) -> tuple[dict[str, Any], bool]:
    pairs: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    saw_none = False
    for line in text.splitlines():
        cleaned = line.strip().strip("-* ").strip()
        if not cleaned:
            continue
        if _clean_atom(cleaned) == "none":
            saw_none = True
            continue
        separator = "=>" if "=>" in cleaned else "->" if "->" in cleaned else ""
        if not separator:
            continue
        left, right = (part.strip() for part in cleaned.split(separator, 1))
        if not left or not right or _clean_atom(left) == _clean_atom(right):
            continue
        key = (left.casefold(), right.casefold())
        if key in seen:
            continue
        seen.add(key)
        pairs.append({"original_surface": left, "alias_surface": right})
    return {"anchor_alias_pairs": pairs[:12]}, bool(pairs) or saw_none


def _calls_by_card(calls: list[NeedCardCall]) -> dict[NeedCardName, NeedCardCall]:
    return {call.card_name: call for call in calls}


def _should_run_search_words_other_language(
    calls: list[NeedCardCall],
    content_language_profile: list[dict[str, Any]],
) -> bool:
    by_card = _calls_by_card(calls)

    def parsed(card_name: NeedCardName) -> dict[str, Any]:
        call = by_card.get(card_name)
        return call.parsed if call is not None and call.parse_valid else {}

    search_words = parsed("search_words").get("anchor_terms") or []
    if not search_words:
        return False
    query_language = parsed("language").get("query_language")
    if not isinstance(query_language, str) or not query_language:
        return False
    target_content_languages = _content_language_codes(content_language_profile)
    if not any(language != query_language for language in target_content_languages):
        return False

    memory_dependence = parsed("memory").get("memory_dependence")
    exact_recall_needed = parsed("exact").get("exact_recall_needed")
    query_type = parsed("shape").get("query_type")
    exact_facets = parsed("facets").get("exact_facets") or []
    return (
        memory_dependence
        in {MemoryDependence.PERSONAL.value, MemoryDependence.MIXED.value}
        or exact_recall_needed is True
        or query_type in {"slot_fill", "broad_list", "temporal"}
        or bool(exact_facets)
    )


def _anchor_aliases_by_search_word(
    alias_pairs: list[dict[str, Any]],
    *,
    anchor_terms: list[str],
    query_language: str | None,
    target_content_languages: list[str],
) -> dict[str, list[dict[str, Any]]]:
    terms_by_key = {term.casefold(): term for term in anchor_terms}
    alias_language = _alias_language_for_search_word(
        query_language,
        target_content_languages,
    )
    alias_kind = "translation" if alias_language is not None else "corpus_surface"
    aliases_by_term: dict[str, list[dict[str, Any]]] = {}
    for pair in alias_pairs:
        original = str(pair.get("original_surface") or "").strip()
        alias_surface = str(pair.get("alias_surface") or "").strip()
        term = terms_by_key.get(original.casefold())
        if term is None or not alias_surface or _clean_atom(term) == _clean_atom(alias_surface):
            continue
        aliases_by_term.setdefault(term, []).append(
            {
                "surface": alias_surface,
                "alias_language": alias_language,
                "alias_kind": alias_kind,
                "confidence": 0.65,
            }
        )
    return {
        term: _normalize_anchor_alias_records(aliases)
        for term, aliases in aliases_by_term.items()
    }


def _alias_language_for_search_word(
    query_language: str | None,
    target_content_languages: list[str],
) -> str | None:
    for language in target_content_languages:
        if language != query_language:
            return language
    return target_content_languages[0] if target_content_languages else None


def _normalize_anchor_alias_records(
    aliases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None, str]] = set()
    for alias in aliases:
        surface = str(alias.get("surface") or "").strip()
        if not surface:
            continue
        language = _language_code_or_none(alias.get("alias_language"))
        alias_kind = _clean_atom(str(alias.get("alias_kind") or ""))
        if alias_kind not in _ALIAS_KIND_VALUES:
            alias_kind = "corpus_surface"
        key = (surface.casefold(), language, alias_kind)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            {
                "surface": surface,
                "alias_language": language,
                "alias_kind": alias_kind,
                "confidence": float(alias.get("confidence") or 0.65),
            }
        )
    return normalized[:2]


def _runtime_anchor_aliases(
    aliases: list[dict[str, Any]],
    *,
    target_content_languages: list[str],
    source: str,
) -> list[RuntimeAnchorAlias]:
    runtime_aliases: list[RuntimeAnchorAlias] = []
    for alias in aliases:
        alias_kind = _clean_atom(str(alias.get("alias_kind") or ""))
        if alias_kind not in _ALIAS_KIND_VALUES:
            alias_kind = "corpus_surface"
        try:
            runtime_aliases.append(
                RuntimeAnchorAlias(
                    surface=str(alias.get("surface") or ""),
                    alias_language=_language_code_or_none(alias.get("alias_language")),
                    alias_kind=alias_kind,
                    confidence=float(alias.get("confidence") or 0.65),
                    derivation={
                        "source": source,
                        "target_content_languages": target_content_languages,
                    },
                )
            )
        except ValueError:
            continue
    return runtime_aliases


def _content_language_codes(content_language_profile: list[dict[str, Any]]) -> list[str]:
    codes: list[str] = []
    seen: set[str] = set()
    for row in content_language_profile:
        code = _language_code_or_none(row.get("language_code"))
        if code is None or code == "unknown" or code in seen:
            continue
        seen.add(code)
        codes.append(code)
    return codes


def _language_code_or_none(value: Any) -> str | None:
    code = _clean_atom(str(value or ""))
    if len(code) == 2 and code.isalpha():
        return code
    return None


def _language_codes_from_output(text: str) -> list[str]:
    normalized = (
        text.strip()
        .replace("<TAB>", "\n")
        .replace("<tab>", "\n")
        .replace("\\t", "\n")
        .replace("\t", "\n")
        .replace("|", "\n")
    )
    codes: list[str] = []
    for line in normalized.splitlines():
        atoms = [_clean_atom(piece) for piece in line.replace(":", " ").split()]
        two_letter_atoms = [
            atom for atom in atoms if len(atom) == 2 and atom.isalpha()
        ]
        if two_letter_atoms:
            codes.append(two_letter_atoms[-1])
    if len(codes) >= 2:
        return codes[:2]
    atoms = [_clean_atom(piece) for piece in normalized.split()]
    for atom in atoms:
        if len(atom) == 2 and atom.isalpha():
            codes.append(atom)
        if len(codes) == 2:
            break
    return codes[:2]


def _parse_yes_no(text: str) -> tuple[bool | None, bool]:
    atom = _first_atom(text)
    if atom in {"yes", "true", "oui", "si", "sí"}:
        return True, True
    if atom in {"no", "false", "non"}:
        return False, True
    return None, False


def _first_atom(text: str) -> str:
    for token in text.replace("\t", " ").split():
        cleaned = _clean_atom(token)
        if cleaned:
            return cleaned
    return ""


def _first_allowed_atom(text: str, allowed: set[str]) -> str:
    for token in text.replace("\t", " ").split():
        cleaned = _clean_atom(token)
        if cleaned in allowed:
            return cleaned
        if ":" in cleaned:
            prefix = cleaned.split(":", 1)[0]
            if prefix in allowed:
                return prefix
    return ""


def _clean_atom(value: str) -> str:
    return value.strip().strip("`*_.,:;[](){}\"'").casefold()


def _normalize_text_list(values: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        normalized.append(text)
    return normalized


def _normalize_exact_facets(values: Any) -> list[ExactFacet]:
    normalized: list[ExactFacet] = []
    seen: set[ExactFacet] = set()
    for value in values:
        try:
            facet = value if isinstance(value, ExactFacet) else ExactFacet(str(value))
        except ValueError:
            mapped = _FACET_MAP.get(_clean_atom(str(value)))
            if mapped is None:
                continue
            facet = mapped
        if facet in seen:
            continue
        seen.add(facet)
        normalized.append(facet)
    return normalized


def _filter_detected_needs(
    needs: list[DetectedNeed],
    resolved_policy: ResolvedRetrievalPolicy,
) -> list[DetectedNeed]:
    allowed_need_types = set(resolved_policy.need_triggers)
    deduped: dict[NeedTrigger, DetectedNeed] = {}
    for need in needs:
        if need.need_type not in allowed_need_types:
            continue
        current = deduped.get(need.need_type)
        if current is None or need.confidence > current.confidence:
            deduped[need.need_type] = need
    return sorted(
        deduped.values(),
        key=lambda need: (-need.confidence, need.need_type.value),
    )


def _authority_context_from_extraction_context(
    context: ExtractionConversationContext,
    *,
    purpose: str,
) -> PromptAuthorityContext:
    return process_authority_context(
        privacy_enforcement=str(getattr(context, "privacy_enforcement", "enforce")),
        user_id=context.user_id,
        privilege_level=getattr(context, "authenticated_user_privilege_level", None),
        is_atagia_master=bool(
            getattr(context, "authenticated_user_is_atagia_master", False)
        ),
        purpose=purpose,
    )
