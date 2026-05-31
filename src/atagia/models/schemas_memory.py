"""Memory domain schemas."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
import re
from typing import Any, Literal, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from atagia.core import json_utils
from atagia.core.language_codes import (
    normalize_iso_639_1_code,
    normalize_optional_iso_639_1_code,
)


class MemoryObjectType(str, Enum):
    EVIDENCE = "evidence"
    BELIEF = "belief"
    INTERACTION_CONTRACT = "interaction_contract"
    STATE_SNAPSHOT = "state_snapshot"
    CONSEQUENCE_CHAIN = "consequence_chain"
    SUMMARY_VIEW = "summary_view"


class MemoryScope(str, Enum):
    # Legacy scopes kept during the additive phase of the namespace redesign.
    # Phase 11 will remove them from the CHECK constraint and enum entries.
    GLOBAL_USER = "global_user"
    ASSISTANT_MODE = "assistant_mode"
    WORKSPACE = "workspace"
    CONVERSATION = "conversation"
    EPHEMERAL_SESSION = "ephemeral_session"
    # Canonical post-redesign scopes (chat / character / user).
    CHAT = "chat"
    CHARACTER = "character"
    USER = "user"


class MemorySensitivity(str, Enum):
    """Per-fact sensitivity gate introduced by the namespace redesign.

    Storage and admin/review paths may carry `unknown`; ordinary retrieval
    treats it as fail-closed (hidden) until a writer assigns a concrete value.
    """

    UNKNOWN = "unknown"
    PUBLIC = "public"
    PRIVATE = "private"
    SECRET = "secret"


class PresenceKind(str, Enum):
    HUMAN = "human"
    OWNED_AI = "owned_ai"
    OWNED_FACET = "owned_facet"
    EXTERNAL_ACTOR = "external_actor"
    OVERSEER = "overseer"
    UNKNOWN = "unknown"


class MindKind(str, Enum):
    HUMAN = "human"
    OWNED_AI = "owned_ai"
    OWNED_FACET = "owned_facet"
    EXTERNAL_ACTOR = "external_actor"
    OVERSEER = "overseer"
    UNKNOWN = "unknown"


class MindTopology(str, Enum):
    UNIMIND = "unimind"
    MULTI_MIND = "multi_mind"
    OJOCENTAURI = "ojocentauri"


class ResponseMode(str, Enum):
    """Per-turn latency/quality tradeoff for context assembly.

    - ``normal``: full synchronous retrieval pipeline (default, byte-identical
      to pre-fast-mode behavior).
    - ``fast``: respond from prepared package + recent transcript + a cheap
      contract read, skipping the retrieval pipeline entirely.
    - ``smart_fast``: respond like ``fast`` immediately, then run the normal
      retrieval in the background so the next turn is warmed.
    """

    NORMAL = "normal"
    FAST = "fast"
    SMART_FAST = "smart_fast"


class OverseerGrantKind(str, Enum):
    READ = "read"
    SUMMARIZE = "summarize"
    COORDINATE = "coordinate"
    AUDIT = "audit"
    RESCOPE = "rescope"


class OverseerGrantTargetKind(str, Enum):
    MIND = "mind"
    SPACE = "space"
    REALM = "realm"


class SpaceBoundaryMode(str, Enum):
    FOCUS = "focus"
    SEVERANCE = "severance"
    PRIVACY_VAULT = "privacy_vault"
    TAGGED = "tagged"


class EmbodimentBoundaryMode(str, Enum):
    NONE = "none"
    ATTRIBUTED = "attributed"
    DIRECT_IF_SAME_BODY = "direct_if_same_body"


class CrossRealmMode(str, Enum):
    NONE = "none"
    ATTRIBUTED = "attributed"
    APPLICABLE = "applicable"


# Canonical post-redesign scopes (retrieval/extraction work after the cutover).
CANONICAL_SCOPES: frozenset[MemoryScope] = frozenset(
    {MemoryScope.CHAT, MemoryScope.CHARACTER, MemoryScope.USER}
)

# Legacy scopes that survive only in storage during the additive phase.
LEGACY_SCOPES: frozenset[MemoryScope] = frozenset(
    {
        MemoryScope.GLOBAL_USER,
        MemoryScope.ASSISTANT_MODE,
        MemoryScope.WORKSPACE,
        MemoryScope.CONVERSATION,
        MemoryScope.EPHEMERAL_SESSION,
    }
)


class MemorySourceKind(str, Enum):
    VERBATIM = "verbatim"
    EXTRACTED = "extracted"
    INFERRED = "inferred"
    SUMMARIZED = "summarized"
    COMPOSED = "composed"


class MemoryCategory(str, Enum):
    PHONE = "phone"
    ADDRESS = "address"
    PIN_OR_PASSWORD = "pin_or_password"
    MEDICATION = "medication"
    FINANCIAL = "financial"
    DATE_OF_BIRTH = "date_of_birth"
    CONTACT_IDENTITY = "contact_identity"
    OTHER_SENSITIVE = "other_sensitive"
    UNKNOWN = "unknown"


class IntimacyBoundary(str, Enum):
    ORDINARY = "ordinary"
    ROMANTIC_PRIVATE = "romantic_private"
    INTIMACY_PRIVATE = "intimacy_private"
    INTIMACY_PREFERENCE_PRIVATE = "intimacy_preference_private"
    INTIMACY_BOUNDARY = "intimacy_boundary"
    AMBIGUOUS_INTIMATE = "ambiguous_intimate"
    SAFETY_BLOCKED = "safety_blocked"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"
    DELETED = "deleted"
    REVIEW_REQUIRED = "review_required"
    PENDING_USER_CONFIRMATION = "pending_user_confirmation"
    DECLINED = "declined"


class MemoryEvidenceSupportKind(str, Enum):
    DIRECT = "direct"
    CONTEXTUAL_DIRECT = "contextual_direct"
    INFERRED = "inferred"
    WEAK_SIGNAL = "weak_signal"


class MemoryEvidencePolarity(str, Enum):
    SUPPORTS = "supports"
    QUALIFIES = "qualifies"
    CONTRADICTS = "contradicts"


class MemoryEvidenceSpanRole(str, Enum):
    SOURCE = "source"
    TRIGGER = "trigger"
    QUALIFIER = "qualifier"
    CONTRADICTION = "contradiction"


class MemoryEvidenceSpeakerRelation(str, Enum):
    SELF_REPORT = "self_report"
    SECOND_PERSON_CONFIRMATION = "second_person_confirmation"
    THIRD_PARTY_REPORT = "third_party_report"
    ASSISTANT_INFERENCE = "assistant_inference"
    BEHAVIORAL_OBSERVATION = "behavioral_observation"
    UNKNOWN = "unknown"


class MemoryEvidenceStatus(str, Enum):
    ACTIVE = "active"
    REVIEW_REQUIRED = "review_required"
    DELETED = "deleted"


class IngestOrigin(str, Enum):
    """Canonical source kind for sidecar message ingestion."""

    LIVE_TURN = "live_turn"
    BACKFILL = "backfill"
    ADMIN_IMPORT = "admin_import"


class ConfirmationStrategy(str, Enum):
    """How extraction may handle memories that require explicit consent."""

    LIVE_PROMPT_ALLOWED = "live_prompt_allowed"
    NEVER_PROMPT_USER = "never_prompt_user"
    ADMIN_REVIEW_ONLY = "admin_review_only"


class MemoryPrivacyMode(str, Enum):
    """User-selected memory trust profile for storage decisions."""

    BALANCED = "balanced"
    TRUSTED_PRIVATE = "trusted_private"


def default_confirmation_strategy_for_origin(
    ingest_origin: IngestOrigin | str | None,
) -> ConfirmationStrategy:
    """Resolve the safe default confirmation strategy for an ingest origin."""

    origin = IngestOrigin(ingest_origin or IngestOrigin.LIVE_TURN.value)
    if origin is IngestOrigin.LIVE_TURN:
        return ConfirmationStrategy.LIVE_PROMPT_ALLOWED
    return ConfirmationStrategy.ADMIN_REVIEW_ONLY


def resolve_confirmation_strategy(
    *,
    ingest_origin: IngestOrigin | str | None,
    confirmation_strategy: ConfirmationStrategy | str | None,
) -> ConfirmationStrategy:
    """Return an explicit confirmation strategy or the origin-derived default."""

    if confirmation_strategy is None:
        return default_confirmation_strategy_for_origin(ingest_origin)
    return ConfirmationStrategy(confirmation_strategy)


def resolve_memory_privacy_mode(
    memory_privacy_mode: MemoryPrivacyMode | str | None,
) -> MemoryPrivacyMode:
    """Return the resolved user memory privacy mode."""

    if memory_privacy_mode is None:
        return MemoryPrivacyMode.BALANCED
    return MemoryPrivacyMode(memory_privacy_mode)


class ConversationStatus(str, Enum):
    ACTIVE = "active"
    CLOSED = "closed"
    ARCHIVED = "archived"
    PENDING_DELETION = "pending_deletion"


class SummaryViewKind(str, Enum):
    CONVERSATION_CHUNK = "conversation_chunk"
    WORKSPACE_ROLLUP = "workspace_rollup"
    CHARACTER_ROLLUP = "character_rollup"
    CONTEXT_VIEW = "context_view"
    EPISODE = "episode"
    THEMATIC_PROFILE = "thematic_profile"


class NeedTrigger(str, Enum):
    AMBIGUITY = "ambiguity"
    CONTRADICTION = "contradiction"
    FOLLOW_UP_FAILURE = "follow_up_failure"
    LOOP = "loop"
    HIGH_STAKES = "high_stakes"
    MODE_SHIFT = "mode_shift"
    FRUSTRATION = "frustration"
    SENSITIVE_CONTEXT = "sensitive_context"
    UNDER_SPECIFIED_REQUEST = "under_specified_request"


class ExactFacet(str, Enum):
    """Categories of exact values an exact-recall query may target."""

    DATE = "date"
    PHONE = "phone"
    EMAIL = "email"
    CODE = "code"
    LOCATION = "location"
    QUANTITY = "quantity"
    PERSON_NAME = "person_name"
    ORG_NAME = "org_name"
    MEDICATION = "medication"
    OTHER_VERBATIM = "other_verbatim"


class VerbatimPinTargetKind(str, Enum):
    """Targets that a verbatim pin may anchor to."""

    MESSAGE = "message"
    MEMORY_OBJECT = "memory_object"
    TEXT_SPAN = "text_span"


class VerbatimPinStatus(str, Enum):
    """Lifecycle states for a verbatim pin."""

    ACTIVE = "active"
    ARCHIVED = "archived"
    EXPIRED = "expired"
    DELETED = "deleted"


class ConsequenceSentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class RetrievalProfileId(str, Enum):
    CODING_DEBUG = "coding_debug"
    RESEARCH_DEEP_DIVE = "research_deep_dive"
    COMPANION = "companion"
    BRAINSTORM = "brainstorm"
    BIOGRAPHICAL_INTERVIEW = "biographical_interview"
    GENERAL_QA = "general_qa"
    INTIMACY = "intimacy"
    PERSONAL_ASSISTANT = "personal_assistant"


TemporalType = Literal["permanent", "bounded", "event_triggered", "ephemeral", "unknown"]
QueryType = Literal["broad_list", "temporal", "slot_fill", "default"]
RawContextAccessMode = Literal["normal", "skipped_raw", "artifact", "verbatim"]
AnswerShape = Literal[
    "single_fact",
    "list",
    "temporal",
    "open_domain",
    "raw_context",
]
CoverageMode = Literal[
    "top_support",
    "exhaustive_known_set",
    "chronology",
    "current_state",
]
SourcePrecision = Literal["required", "preferred"]
EvidenceCoverageState = Literal[
    "unknown",
    "complete",
    "partial",
    "conflicting",
    "insufficient",
]

_VALID_TEMPORAL_TYPES = frozenset(get_args(TemporalType))
_QUERY_HINT_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
_FTS5_OPERATOR_TOKENS = frozenset({"and", "or", "not", "near"})
_EXPLICIT_TEMPORAL_FIELDS_DEFAULT_CONFIDENCE = 0.6


def _ensure_unique_values(values: list[Any]) -> list[Any]:
    """Reject duplicate values while preserving order semantics."""
    if len(values) != len(set(values)):
        raise ValueError("List values must be unique")
    return values


def _searchable_hint_tokens(value: str) -> tuple[str, ...]:
    tokens: list[str] = []
    seen: set[str] = set()
    for token in _QUERY_HINT_TOKEN_PATTERN.findall(value.lower()):
        normalized = token.strip("_")
        if not normalized:
            continue
        if not any(character.isalnum() for character in normalized):
            continue
        if normalized in _FTS5_OPERATOR_TOKENS:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        tokens.append(normalized)
    return tuple(tokens)


def _append_unique_text(target: list[str], value: str | None) -> None:
    if value is None:
        return
    normalized = value.strip()
    if not normalized or normalized in target:
        return
    target.append(normalized)


def derive_answer_coverage_fields(
    *,
    query_type: QueryType,
    exact_recall_needed: bool = False,
    raw_context_access_mode: RawContextAccessMode = "normal",
) -> tuple[AnswerShape, CoverageMode, SourcePrecision]:
    """Derive lean answer-support routing from existing query intelligence."""
    if raw_context_access_mode in {"artifact", "verbatim"}:
        return ("raw_context", "top_support", "required")
    if query_type == "broad_list":
        return ("list", "exhaustive_known_set", "required")
    if query_type == "temporal":
        return ("temporal", "chronology", "required")
    if query_type == "slot_fill":
        coverage_mode: CoverageMode = "current_state" if exact_recall_needed else "top_support"
        return ("single_fact", coverage_mode, "required")
    return ("open_domain", "top_support", "preferred")


def _normalize_sparse_hint_precision_fields(
    hint: "SparseQueryHint",
    *,
    query_type: QueryType,
    callback_bias: bool,
) -> "SparseQueryHint":
    quoted_phrases: list[str] = []
    must_keep_terms: list[str] = []

    for phrase in hint.quoted_phrases:
        if len(_searchable_hint_tokens(phrase)) <= 1:
            _append_unique_text(must_keep_terms, phrase)
            continue
        _append_unique_text(quoted_phrases, phrase)

    for term in hint.must_keep_terms:
        _append_unique_text(must_keep_terms, term)
        if len(_searchable_hint_tokens(term)) >= 2:
            _append_unique_text(quoted_phrases, term)

    should_promote_fts_anchor = (
        not quoted_phrases
        and not must_keep_terms
        and hint.fts_phrase is not None
        and bool(_searchable_hint_tokens(hint.fts_phrase))
        and (callback_bias or query_type in {"slot_fill", "broad_list"})
    )
    if should_promote_fts_anchor:
        if len(_searchable_hint_tokens(hint.fts_phrase)) >= 2:
            _append_unique_text(quoted_phrases, hint.fts_phrase)
        else:
            _append_unique_text(must_keep_terms, hint.fts_phrase)

    return hint.model_copy(
        update={
            "quoted_phrases": quoted_phrases,
            "must_keep_terms": must_keep_terms,
        }
    )


def _sparse_hint_signature(hint: "SparseQueryHint") -> tuple[str, ...]:
    seen: set[str] = set()
    values: list[str] = []
    if hint.fts_phrase is not None:
        values.append(hint.fts_phrase)
    values.extend(hint.quoted_phrases)
    values.extend(hint.must_keep_terms)
    for value in values:
        for token in _searchable_hint_tokens(value):
            if token in seen:
                continue
            seen.add(token)
    return tuple(sorted(seen))


class RetrievalParams(BaseModel):
    """Retrieval tuning knobs carried by an assistant mode policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fts_limit: int = Field(ge=0)
    vector_limit: int = Field(ge=0)
    graph_hops: int = Field(ge=0)
    rerank_top_k: int = Field(gt=0)
    final_context_items: int = Field(gt=0)


class OperationalPower(str, Enum):
    NORMAL = "normal"
    CONSTRAINED = "constrained"
    CRITICAL = "critical"


class OperationalConnectivity(str, Enum):
    ONLINE = "online"
    DEGRADED = "degraded"
    OFFLINE = "offline"


class OperationalSafety(str, Enum):
    NORMAL = "normal"
    HIGH_STAKES = "high_stakes"
    EMERGENCY = "emergency"


class OperationalIncidentScope(str, Enum):
    NONE = "none"
    LOCAL_DISRUPTION = "local_disruption"
    DISASTER = "disaster"


class OperationalCompute(str, Enum):
    REMOTE_ALLOWED = "remote_allowed"
    LOCAL_PREFERRED = "local_preferred"
    LOCAL_ONLY = "local_only"


class OperationalSignals(BaseModel):
    """Structured operational signals supplied by a client/device."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    power: OperationalPower | None = None
    connectivity: OperationalConnectivity | None = None
    safety: OperationalSafety | None = None
    incident_scope: OperationalIncidentScope | None = None
    compute: OperationalCompute | None = None


class OperationalRetrievalParamsOverride(BaseModel):
    """Operational restriction for retrieval knobs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fts_limit: int | None = Field(default=None, ge=0)
    vector_limit: int | None = Field(default=None, ge=0)
    graph_hops: int | None = Field(default=None, ge=0)
    rerank_top_k: int | None = Field(default=None, gt=0)
    final_context_items: int | None = Field(default=None, gt=0)


class OperationalPolicyOverride(BaseModel):
    """Fields an operational profile may restrict in Phase 0."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed_scopes: list[MemoryScope] | None = None
    preferred_memory_types: list[MemoryObjectType] | None = None
    need_triggers: list[NeedTrigger] | None = None
    contract_dimensions_priority: list[str] | None = None
    context_budget_tokens: int | None = Field(default=None, gt=0)
    transcript_budget_tokens: int | None = Field(default=None, gt=0)
    retrieval_params: OperationalRetrievalParamsOverride | None = None

    @field_validator("allowed_scopes")
    @classmethod
    def validate_allowed_scopes(cls, values: list[MemoryScope] | None) -> list[MemoryScope] | None:
        return None if values is None else _ensure_unique_values(values)

    @field_validator("preferred_memory_types")
    @classmethod
    def validate_preferred_memory_types(
        cls,
        values: list[MemoryObjectType] | None,
    ) -> list[MemoryObjectType] | None:
        return None if values is None else _ensure_unique_values(values)

    @field_validator("need_triggers")
    @classmethod
    def validate_need_triggers(cls, values: list[NeedTrigger] | None) -> list[NeedTrigger] | None:
        return None if values is None else _ensure_unique_values(values)

    @field_validator("contract_dimensions_priority")
    @classmethod
    def validate_contract_dimensions_priority(cls, values: list[str] | None) -> list[str] | None:
        if values is None:
            return None
        if any(not value.strip() for value in values):
            raise ValueError("Contract dimensions must be non-empty strings")
        return _ensure_unique_values(values)

    def to_policy_override_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


class OperationalRiskLevel(str, Enum):
    NORMAL = "normal"
    SENSITIVE = "sensitive"
    HIGH_RISK = "high_risk"


class OperationalProfile(BaseModel):
    """Canonical operational profile preset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operational_profile_id: str = Field(min_length=1, max_length=64)
    display_name: str = Field(min_length=1, max_length=128)
    risk_level: OperationalRiskLevel
    signals: OperationalSignals
    policy_override: OperationalPolicyOverride = Field(default_factory=OperationalPolicyOverride)
    profile_hash: str | None = Field(default=None, exclude=True)


class OperationalProfileSnapshot(BaseModel):
    """Stable operational profile shape carried by cache entries and jobs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str = Field(min_length=1, max_length=64)
    signals: OperationalSignals
    risk_level: OperationalRiskLevel
    authorized: bool
    profile_hash: str = Field(min_length=1)
    token: str = Field(min_length=1)


class ResolvedOperationalProfile(BaseModel):
    """Resolved operational profile with effective signals and policy overlay."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot: OperationalProfileSnapshot
    policy_override: OperationalPolicyOverride = Field(default_factory=OperationalPolicyOverride)


class ContextCachePolicy(BaseModel):
    """Adaptive context-cache settings attached to a retrieval profile."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_ttl_seconds: int = Field(gt=0)
    sync_threshold: float = Field(gt=0.0, le=1.0)
    max_messages_without_refresh: int = Field(ge=1)
    max_minutes_without_refresh: int = Field(ge=1)


class RetrievalProfileManifest(BaseModel):
    """Validated retrieval profile loaded from JSON files."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: RetrievalProfileId
    display_name: str = Field(min_length=1)
    allow_intimacy_context: bool = False
    preferred_memory_types: list[MemoryObjectType] = Field(min_length=1)
    need_triggers: list[NeedTrigger] = Field(default_factory=list)
    contract_dimensions_priority: list[str] = Field(min_length=1)
    privacy_ceiling: int = Field(ge=0, le=3)
    context_budget_tokens: int = Field(gt=0)
    transcript_budget_tokens: int = Field(gt=0)
    retrieval_params: RetrievalParams
    context_cache_policy: ContextCachePolicy
    prompt_hash: str | None = Field(default=None, exclude=True)

    @field_validator("preferred_memory_types")
    @classmethod
    def validate_preferred_memory_types(
        cls,
        values: list[MemoryObjectType],
    ) -> list[MemoryObjectType]:
        return _ensure_unique_values(values)

    @field_validator("need_triggers")
    @classmethod
    def validate_need_triggers(cls, values: list[NeedTrigger]) -> list[NeedTrigger]:
        return _ensure_unique_values(values)

    @field_validator("contract_dimensions_priority")
    @classmethod
    def validate_contract_dimensions_priority(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("Contract dimensions must be non-empty strings")
        return _ensure_unique_values(values)


class ExtractionContextMessage(BaseModel):
    """Recent conversation message passed into the extractor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str | None = None
    role: str
    content: str
    seq: int | None = None
    occurred_at: str | None = None


class ExtractionConversationContext(BaseModel):
    """Execution context for a single extraction run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str
    conversation_id: str
    source_message_id: str
    workspace_id: str | None = None
    assistant_mode_id: str
    user_persona_id: str | None = None
    platform_id: str = "default"
    character_id: str | None = None
    active_presence_id: str | None = None
    active_presence_kind: PresenceKind = PresenceKind.UNKNOWN
    active_presence_display_name: str | None = None
    source_presence_id: str | None = None
    source_presence_kind: PresenceKind = PresenceKind.UNKNOWN
    source_presence_display_name: str | None = None
    active_space_id: str | None = None
    active_space_boundary_mode: SpaceBoundaryMode = SpaceBoundaryMode.FOCUS
    active_space_display_name: str | None = None
    active_mind_id: str | None = None
    source_mind_id: str | None = None
    active_mind_display_name: str | None = None
    mind_topology: MindTopology = MindTopology.UNIMIND
    active_embodiment_id: str | None = None
    active_embodiment_display_name: str | None = None
    cross_embodiment_mode: EmbodimentBoundaryMode = (
        EmbodimentBoundaryMode.DIRECT_IF_SAME_BODY
    )
    active_realm_id: str | None = None
    active_realm_display_name: str | None = None
    cross_realm_mode: CrossRealmMode = CrossRealmMode.NONE
    mode: str | None = None
    recent_messages: list[ExtractionContextMessage] = Field(default_factory=list)
    temporary: bool = False
    temporary_ttl_seconds: int | None = None
    purge_on_close: bool = False
    isolated_mode: bool = False
    incognito: bool = False
    remember_across_chats: bool = True
    remember_across_devices: bool = True
    ingest_origin: IngestOrigin = IngestOrigin.LIVE_TURN
    confirmation_strategy: ConfirmationStrategy | None = None
    memory_privacy_mode: MemoryPrivacyMode = MemoryPrivacyMode.BALANCED
    privacy_enforcement: Literal["enforce", "audit_only", "off"] = "enforce"
    authenticated_user_privilege_level: str | None = None
    authenticated_user_is_atagia_master: bool = False

    @model_validator(mode="before")
    @classmethod
    def resolve_confirmation_strategy_default(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if normalized.get("confirmation_strategy") is None:
            normalized["confirmation_strategy"] = resolve_confirmation_strategy(
                ingest_origin=normalized.get("ingest_origin"),
                confirmation_strategy=None,
            )
        return normalized


class ExtractedMemoryBase(BaseModel):
    """Shared fields returned by the extraction model."""

    model_config = ConfigDict(extra="ignore")

    canonical_text: str = Field(min_length=1)
    index_text: str | None = None
    scope: MemoryScope
    confidence: float = Field(ge=0.0, le=1.0)
    source_kind: MemorySourceKind
    support_kind: MemoryEvidenceSupportKind | None = None
    evidence_polarity: MemoryEvidencePolarity | None = None
    speaker_relation_to_subject: MemoryEvidenceSpeakerRelation | None = None
    source_quote: str | None = None
    trigger_message_ids: list[str] = Field(default_factory=list)
    trigger_quote: str | None = None
    support_rationale: str | None = None
    confidence_details: dict[str, Any] = Field(default_factory=dict)
    privacy_level: int = Field(ge=0, le=3)
    sensitivity: MemorySensitivity = MemorySensitivity.UNKNOWN
    themes: list[str] = Field(default_factory=list)
    auto_expires: bool = False
    platform_locked: bool = False
    platform_lock_reason: str | None = None
    memory_category: MemoryCategory = MemoryCategory.UNKNOWN
    intimacy_boundary: IntimacyBoundary = IntimacyBoundary.ORDINARY
    intimacy_boundary_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    preserve_verbatim: bool = False
    informational_mention: bool | None = None
    subject_presence_ids: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
    temporal_type: TemporalType = "unknown"
    valid_from_iso: str | None = None
    valid_to_iso: str | None = None
    temporal_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    language_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if "canonical_text" not in normalized:
            for field_name in (
                "text",
                "memory_text",
                "summary_text",
                "description",
                "content",
            ):
                value = normalized.get(field_name)
                if isinstance(value, str) and value.strip():
                    normalized["canonical_text"] = value
                    break
        if "confidence" not in normalized and "score" in normalized:
            normalized["confidence"] = normalized.get("score")
        normalized.setdefault("confidence", 0.5)
        normalized.setdefault("scope", MemoryScope.CONVERSATION.value)
        normalized.setdefault("source_kind", MemorySourceKind.EXTRACTED.value)
        normalized.setdefault("trigger_message_ids", [])
        normalized.setdefault("confidence_details", {})
        normalized.setdefault("privacy_level", 0)
        normalized.setdefault("sensitivity", MemorySensitivity.UNKNOWN.value)
        normalized.setdefault("themes", [])
        normalized.setdefault("auto_expires", False)
        normalized.setdefault("platform_locked", False)
        normalized.setdefault("intimacy_boundary", IntimacyBoundary.ORDINARY.value)
        normalized.setdefault("intimacy_boundary_confidence", 0.0)
        normalized.setdefault("payload", {})
        normalized.setdefault("temporal_type", "unknown")
        if "temporal_confidence" not in normalized:
            has_temporal_bounds = bool(normalized.get("valid_from_iso") or normalized.get("valid_to_iso"))
            temporal_type = str(normalized.get("temporal_type") or "unknown").strip().lower()
            normalized["temporal_confidence"] = (
                _EXPLICIT_TEMPORAL_FIELDS_DEFAULT_CONFIDENCE
                if has_temporal_bounds or temporal_type not in {"", "unknown"}
                else 0.0
            )
        return normalized

    @field_validator("language_codes", mode="before")
    @classmethod
    def normalize_language_codes_before_type_validation(cls, value: Any) -> list[str]:
        values = value if isinstance(value, list) else [value]
        normalized: list[str] = []
        seen: set[str] = set()
        for item in values:
            code = normalize_optional_iso_639_1_code(item)
            if code is None or code in seen:
                continue
            seen.add(code)
            normalized.append(code)
        return normalized

    @field_validator("index_text")
    @classmethod
    def validate_index_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("source_quote", "trigger_quote", "support_rationale")
    @classmethod
    def validate_optional_evidence_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split())
        return normalized or None

    @field_validator("trigger_message_ids")
    @classmethod
    def normalize_trigger_message_ids(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = str(value).strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            normalized.append(cleaned)
        return normalized

    @field_validator("themes")
    @classmethod
    def validate_themes(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            theme = str(item).strip()
            if not theme:
                continue
            key = theme.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(theme)
        return normalized

    @field_validator("platform_lock_reason")
    @classmethod
    def validate_platform_lock_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("subject_presence_ids")
    @classmethod
    def normalize_subject_presence_ids(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = str(value).strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            normalized.append(cleaned)
        return normalized

    @field_validator("valid_from_iso", "valid_to_iso")
    @classmethod
    def validate_temporal_iso(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        datetime.fromisoformat(normalized)
        return normalized

    @model_validator(mode="after")
    def validate_temporal_bounds(self) -> "ExtractedMemoryBase":
        if self.valid_from_iso and self.valid_to_iso:
            if datetime.fromisoformat(self.valid_from_iso) > datetime.fromisoformat(self.valid_to_iso):
                raise ValueError("valid_from_iso must be <= valid_to_iso")
        return self

    @model_validator(mode="after")
    def validate_language_codes(self) -> "ExtractedMemoryBase":
        if not self.language_codes:
            raise ValueError(
                "language_codes must contain at least one ISO 639-1 code for the language of canonical_text"
            )
        seen: set[str] = set()
        for code in self.language_codes:
            lowered = normalize_iso_639_1_code(code)
            seen.add(lowered)
        self.language_codes = sorted(seen)
        return self


class ExtractedEvidence(ExtractedMemoryBase):
    """Grounded observation extracted from a message."""


class ExtractedBelief(ExtractedMemoryBase):
    """Belief candidate inferred from the message."""

    claim_key: str = Field(min_length=1)
    claim_value: str = Field(min_length=1)


class ExtractedContractSignal(ExtractedMemoryBase):
    """Collaboration preference signal extracted from the message."""


class ExtractedStateUpdate(ExtractedMemoryBase):
    """State snapshot update extracted from the message."""


class ExtractionResult(BaseModel):
    """Structured output returned by the memory extractor."""

    model_config = ConfigDict(extra="ignore")

    evidences: list[ExtractedEvidence] = Field(default_factory=list)
    beliefs: list[ExtractedBelief] = Field(default_factory=list)
    contract_signals: list[ExtractedContractSignal] = Field(default_factory=list)
    state_updates: list[ExtractedStateUpdate] = Field(default_factory=list)
    # Hook for mode-shift detection in Step 7.
    mode_guess: str | None = None
    nothing_durable: bool = False

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_payload(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        used_legacy_container = False
        metadata = normalized.pop("extraction_metadata", None)
        normalized.pop("extraction_rationale", None)
        normalized.pop("rationale", None)
        normalized.pop("extraction_id", None)
        normalized.pop("source_message_timestamp", None)
        evidences = list(normalized.get("evidences", []))
        beliefs: list[Any] = []
        for item in normalized.get("beliefs", []):
            if isinstance(item, dict) and (item.get("claim_key") is None or item.get("claim_value") is None):
                evidences.append(dict(item))
                continue
            beliefs.append(item)
        contract_signals = list(normalized.get("contract_signals", []))
        state_updates = list(normalized.get("state_updates", []))

        legacy_items = None
        for field_name in (
            "durable_memory_items",
            "durable_memories",
            "extracted_items",
            "items",
        ):
            if field_name in normalized:
                legacy_items = normalized.pop(field_name)
                used_legacy_container = True
                break

        if legacy_items is not None:
            for item in legacy_items if isinstance(legacy_items, list) else []:
                if not isinstance(item, dict):
                    continue
                normalized_item = dict(item)
                item_type = (
                    str(
                        normalized_item.get("memory_type")
                        or normalized_item.get("item_type")
                        or normalized_item.get("object_type")
                        or normalized_item.get("type")
                        or normalized_item.get("kind")
                        or ""
                    )
                    .strip()
                    .lower()
                )
                if ("claim_key" in normalized_item and "claim_value" in normalized_item) or (
                    item_type == MemoryObjectType.BELIEF.value
                    and normalized_item.get("claim_key") is not None
                    and normalized_item.get("claim_value") is not None
                ):
                    claim_key = str(normalized_item.get("claim_key") or "").strip()
                    claim_value_raw = normalized_item.get("claim_value")
                    if isinstance(claim_value_raw, str):
                        claim_value = claim_value_raw.strip()
                    elif claim_value_raw is None:
                        claim_value = ""
                    else:
                        claim_value = json_utils.dumps(claim_value_raw, sort_keys=True)
                    if claim_key and claim_value:
                        normalized_item["claim_key"] = claim_key
                        normalized_item["claim_value"] = claim_value
                        beliefs.append(normalized_item)
                    else:
                        evidences.append(normalized_item)
                elif item_type in {
                    MemoryObjectType.INTERACTION_CONTRACT.value,
                    "contract",
                    "contract_signal",
                }:
                    contract_signals.append(normalized_item)
                elif item_type in {
                    MemoryObjectType.STATE_SNAPSHOT.value,
                    "state",
                    "state_update",
                }:
                    state_updates.append(normalized_item)
                else:
                    evidences.append(normalized_item)
        normalized["evidences"] = evidences
        normalized["beliefs"] = beliefs
        normalized["contract_signals"] = contract_signals
        normalized["state_updates"] = state_updates
        if used_legacy_container:
            for field_name in (
                "evidences",
                "beliefs",
                "contract_signals",
                "state_updates",
            ):
                normalized[field_name] = [
                    (
                        {
                            **item,
                            "language_codes": ["en"],
                        }
                        if isinstance(item, dict) and "canonical_text" in item and "language_codes" not in item
                        else item
                    )
                    for item in normalized[field_name]
                ]

        if isinstance(metadata, dict):
            if "nothing_durable" not in normalized and "nothing_durable" in metadata:
                normalized["nothing_durable"] = bool(metadata["nothing_durable"])
            if normalized.get("mode_guess") is None and isinstance(metadata.get("mode_guess"), str):
                normalized["mode_guess"] = metadata["mode_guess"]
        if evidences or beliefs or contract_signals or state_updates:
            normalized["nothing_durable"] = False
        return normalized

    @model_validator(mode="after")
    def validate_nothing_durable_consistency(self) -> "ExtractionResult":
        if self.nothing_durable and (self.evidences or self.beliefs or self.contract_signals or self.state_updates):
            raise ValueError("nothing_durable=true but extraction lists are non-empty")
        return self


LeanCandidateKind = Literal["evidence", "belief", "contract_signal", "state_update"]
LeanSubjectScope = Literal["chat", "character", "user"]


class LeanTemporalStatus(BaseModel):
    """Compact temporal annotation produced by the lean extraction contract."""

    model_config = ConfigDict(extra="ignore")

    type: TemporalType = "unknown"
    valid_from_iso: str | None = None
    valid_to_iso: str | None = None

    @field_validator("valid_from_iso", "valid_to_iso")
    @classmethod
    def validate_temporal_iso(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        datetime.fromisoformat(normalized)
        return normalized

    @model_validator(mode="after")
    def validate_temporal_bounds(self) -> "LeanTemporalStatus":
        if self.valid_from_iso and self.valid_to_iso:
            if datetime.fromisoformat(self.valid_from_iso) > datetime.fromisoformat(self.valid_to_iso):
                raise ValueError("valid_from_iso must be <= valid_to_iso")
        return self


class LeanExtractionCandidate(BaseModel):
    """Single durable memory candidate in the model-facing lean contract.

    The extractor maps these into the rich ``Extracted*`` objects server-side;
    everything the model no longer produces receives an approved default there.
    """

    model_config = ConfigDict(extra="ignore")

    canonical_text: str = Field(min_length=1)
    kind: LeanCandidateKind
    subject_scope: LeanSubjectScope
    confidence: float = Field(ge=0.0, le=1.0)
    language_codes: list[str]
    index_text: str | None = None
    preserve_verbatim: bool = False
    source_span: str | None = None
    temporal_status: LeanTemporalStatus | None = None
    support_kind: MemoryEvidenceSupportKind = MemoryEvidenceSupportKind.DIRECT
    claim_key: str | None = None
    claim_value: str | None = None

    @field_validator("language_codes", mode="before")
    @classmethod
    def normalize_language_codes_before_type_validation(cls, value: Any) -> list[str]:
        values = value if isinstance(value, list) else [value]
        normalized: list[str] = []
        seen: set[str] = set()
        for item in values:
            code = normalize_optional_iso_639_1_code(item)
            if code is None or code in seen:
                continue
            seen.add(code)
            normalized.append(code)
        return normalized

    @field_validator("index_text")
    @classmethod
    def validate_index_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("source_span")
    @classmethod
    def validate_source_span(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split())
        return normalized or None

    @model_validator(mode="after")
    def validate_language_codes(self) -> "LeanExtractionCandidate":
        if not self.language_codes:
            raise ValueError(
                "language_codes must contain at least one ISO 639-1 code for the language of canonical_text"
            )
        seen: set[str] = set()
        for code in self.language_codes:
            seen.add(normalize_iso_639_1_code(code))
        self.language_codes = sorted(seen)
        return self

    @model_validator(mode="after")
    def validate_belief_claim(self) -> "LeanExtractionCandidate":
        if self.kind == "belief":
            claim_key = (self.claim_key or "").strip()
            claim_value = (self.claim_value or "").strip()
            if not claim_key or not claim_value:
                raise ValueError("belief candidates require non-empty claim_key and claim_value")
            self.claim_key = claim_key
            self.claim_value = claim_value
        return self


class LeanExtractionResult(BaseModel):
    """Lean, model-facing structured output for memory extraction."""

    model_config = ConfigDict(extra="ignore")

    nothing_durable: bool = False
    candidates: list[LeanExtractionCandidate] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_nothing_durable_consistency(self) -> "LeanExtractionResult":
        if self.nothing_durable and self.candidates:
            raise ValueError("nothing_durable=true but candidates are non-empty")
        return self


class ContractSignal(BaseModel):
    """Contract preference signal extracted from a message."""

    model_config = ConfigDict(extra="ignore")

    canonical_text: str = Field(min_length=1)
    dimension_name: str = Field(min_length=1)
    value_json: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    scope: MemoryScope
    source_kind: MemorySourceKind = MemorySourceKind.INFERRED
    privacy_level: int = Field(ge=0, le=3, default=1)
    language_codes: list[str] = Field(default_factory=list)

    @field_validator("dimension_name")
    @classmethod
    def validate_dimension_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("dimension_name must be non-empty")
        return normalized

    @field_validator("language_codes")
    @classmethod
    def validate_language_codes(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            code = _normalize_profile_language_code(value)
            if code in seen:
                continue
            seen.add(code)
            normalized.append(code)
        return normalized


class ContractProjectionResult(BaseModel):
    """Structured output returned by the contract projector."""

    model_config = ConfigDict(extra="ignore")

    signals: list[ContractSignal] = Field(default_factory=list)
    nothing_durable: bool = False

    @staticmethod
    def _normalize_signal_payload(value: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(value)
        signals = []
        for signal in normalized["signals"]:
            if isinstance(signal, dict) and signal.get("nothing_durable") is True:
                continue
            if not isinstance(signal, dict):
                continue
            if not signal.get("canonical_text") or not signal.get("dimension_name") or not signal.get("scope"):
                continue
            normalized_signal = dict(signal)
            if not isinstance(normalized_signal.get("value_json"), dict):
                aliased_value = _first_present_contract_signal_value(
                    normalized_signal,
                    ("signal_value", "preference", "value", "extracted_value"),
                )
                if isinstance(aliased_value, dict):
                    normalized_signal["value_json"] = aliased_value
                elif aliased_value is not None:
                    normalized_signal["value_json"] = {"label": str(aliased_value)}
            signals.append(normalized_signal)
        normalized["signals"] = signals
        if signals:
            normalized["nothing_durable"] = False
        else:
            normalized["nothing_durable"] = True
        return normalized

    @model_validator(mode="before")
    @classmethod
    def normalize_root_list(cls, value: Any) -> Any:
        if isinstance(value, list):
            return cls._normalize_signal_payload({"signals": value, "nothing_durable": not value})
        if isinstance(value, dict) and isinstance(value.get("signals"), list):
            return cls._normalize_signal_payload(value)
        return value

    @model_validator(mode="after")
    def validate_nothing_durable_consistency(self) -> "ContractProjectionResult":
        if self.nothing_durable and self.signals:
            raise ValueError("nothing_durable=true but contract signals are non-empty")
        return self


def _first_present_contract_signal_value(
    signal: dict[str, Any],
    keys: tuple[str, ...],
) -> Any | None:
    for key in keys:
        if key in signal:
            return signal[key]
    return None


class DetectedNeed(BaseModel):
    """Need signal detected from the active message and recent context."""

    model_config = ConfigDict(extra="ignore")

    need_type: NeedTrigger
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1)

    @field_validator("reasoning")
    @classmethod
    def validate_reasoning(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("reasoning must be non-empty")
        return normalized


class ConsequenceSignal(BaseModel):
    """Detected report of an outcome caused by prior assistant action."""

    model_config = ConfigDict(extra="ignore")

    is_consequence: bool
    action_description: str = ""
    outcome_description: str = ""
    outcome_sentiment: ConsequenceSentiment = ConsequenceSentiment.NEUTRAL
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    likely_action_message_id: str | None = None
    language_codes: list[str] = Field(default_factory=list)

    @field_validator("action_description", "outcome_description", mode="before")
    @classmethod
    def normalize_nullable_descriptions(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value)

    @field_validator("outcome_sentiment", mode="before")
    @classmethod
    def normalize_nullable_sentiment(cls, value: Any) -> Any:
        if value is None:
            return ConsequenceSentiment.NEUTRAL
        return value

    @field_validator("language_codes")
    @classmethod
    def validate_language_codes(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            code = _normalize_profile_language_code(value)
            if code in seen:
                continue
            seen.add(code)
            normalized.append(code)
        return normalized

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_nullable_confidence(cls, value: Any) -> Any:
        if value is None:
            return 0.0
        return value

    @field_validator("action_description", "outcome_description")
    @classmethod
    def validate_descriptions(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def validate_consequence_confidence(self) -> "ConsequenceSignal":
        if self.is_consequence and self.confidence < 0.1:
            raise ValueError("is_consequence=true requires confidence >= 0.1")
        return self


class ConsequenceChainResult(BaseModel):
    """Identifiers returned after building a consequence chain."""

    model_config = ConfigDict(extra="forbid")

    chain_id: str
    action_memory_id: str
    outcome_memory_id: str
    tendency_belief_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class TemporalQueryRange(BaseModel):
    """Normalized temporal range derived from a user query."""

    model_config = ConfigDict(extra="ignore")

    start: datetime
    end: datetime

    @model_validator(mode="after")
    def validate_bounds(self) -> "TemporalQueryRange":
        if self.end < self.start:
            raise ValueError("TemporalQueryRange.end must be >= start")
        return self


class SparseQueryHint(BaseModel):
    """Sparse lexical shaping hints for one semantic sub-query."""

    model_config = ConfigDict(extra="ignore")

    sub_query_text: str = Field(min_length=1)
    fts_phrase: str | None = None
    quoted_phrases: list[str] = Field(default_factory=list)
    must_keep_terms: list[str] = Field(default_factory=list)

    @field_validator("sub_query_text", "fts_phrase")
    @classmethod
    def validate_optional_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("quoted_phrases", "must_keep_terms")
    @classmethod
    def validate_sparse_lists(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values if value.strip()]
        return _ensure_unique_values(normalized)

    @model_validator(mode="after")
    def validate_not_empty(self) -> "SparseQueryHint":
        if self.fts_phrase is None and not self.quoted_phrases and not self.must_keep_terms:
            raise ValueError("SparseQueryHint requires at least one of fts_phrase, quoted_phrases, or must_keep_terms")
        return self


AnchorType = Literal[
    "proper_name",
    "person",
    "organization",
    "location",
    "code",
    "quantity",
    "date_time",
    "address",
    "quoted_phrase",
    "attribute",
    "concept",
    "unknown",
]

AliasKind = Literal[
    "translation",
    "transliteration",
    "spelling_variant",
    "acronym_expansion",
    "domain_synonym",
    "corpus_surface",
]

_PRESERVE_VERBATIM_ANCHOR_TYPES: set[str] = {
    "proper_name",
    "person",
    "organization",
    "location",
    "code",
    "quantity",
    "date_time",
    "address",
    "quoted_phrase",
}


class RuntimeAnchorAlias(BaseModel):
    """Runtime alias surface for retrieval only; never evidential proof."""

    model_config = ConfigDict(extra="ignore")

    surface: str = Field(min_length=1)
    alias_language: str | None = None
    alias_kind: AliasKind
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    derivation: dict[str, Any] = Field(default_factory=dict)
    non_evidential: bool = True

    @field_validator("surface")
    @classmethod
    def validate_required_surface(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("surface must be non-empty")
        return normalized

    @field_validator("alias_language")
    @classmethod
    def validate_optional_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("non_evidential")
    @classmethod
    def validate_non_evidential(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("runtime aliases must be non_evidential")
        return value


class RuntimeAnchor(BaseModel):
    """Structured query anchor for runtime retrieval planning only."""

    model_config = ConfigDict(extra="ignore")

    sub_query_text: str = Field(min_length=1)
    anchor_type: AnchorType = "unknown"
    original_surface: str = Field(min_length=1)
    normalized_surface: str | None = None
    preserve_verbatim: bool = False
    aliases: list[RuntimeAnchorAlias] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    derivation: dict[str, Any] = Field(default_factory=dict)
    non_evidential: bool = True

    @field_validator("sub_query_text")
    @classmethod
    def validate_required_sub_query_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("sub_query_text must be non-empty")
        return normalized

    @field_validator("original_surface")
    @classmethod
    def validate_required_original_surface(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("original_surface must be non-empty")
        return normalized

    @field_validator("normalized_surface")
    @classmethod
    def validate_optional_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("aliases")
    @classmethod
    def validate_unique_aliases(
        cls,
        values: list[RuntimeAnchorAlias],
    ) -> list[RuntimeAnchorAlias]:
        seen: set[tuple[str, str | None, str]] = set()
        normalized: list[RuntimeAnchorAlias] = []
        for alias in values:
            signature = (alias.surface, alias.alias_language, alias.alias_kind)
            if signature in seen:
                continue
            seen.add(signature)
            normalized.append(alias)
        return normalized

    @field_validator("non_evidential")
    @classmethod
    def validate_non_evidential(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("runtime anchors must be non_evidential")
        return value

    @model_validator(mode="after")
    def validate_preserve_verbatim(self) -> "RuntimeAnchor":
        if self.anchor_type in _PRESERVE_VERBATIM_ANCHOR_TYPES and not self.preserve_verbatim:
            raise ValueError(
                "literal/name/code anchors must set preserve_verbatim=true"
            )
        return self


class TemporaryScaffoldingTrace(BaseModel):
    """Trace-only label for temporary recovery mechanisms."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["temporary_scaffolding"] = "temporary_scaffolding"
    component: str
    mechanism: str
    trace_flag: str
    intended_metric: str
    replacement_architecture: str
    retirement_condition: str


class QueryIntelligenceResult(BaseModel):
    """Structured query understanding returned by need detection."""

    model_config = ConfigDict(extra="ignore")

    _temporary_scaffolding: list[TemporaryScaffoldingTrace] = PrivateAttr(
        default_factory=list
    )

    needs: list[DetectedNeed] = Field(default_factory=list)
    temporal_range: TemporalQueryRange | None = None
    sub_queries: list[str] = Field(min_length=1, max_length=3)
    callback_bias: bool = False
    raw_context_access_mode: RawContextAccessMode = "normal"
    sparse_query_hints: list[SparseQueryHint] = Field(default_factory=list)
    query_language: str | None = None
    answer_language: str | None = None
    anchors: list[RuntimeAnchor] = Field(default_factory=list)
    query_type: QueryType = "default"
    retrieval_levels: list[int] = Field(default_factory=lambda: [0])
    # Wave 1 batch 2 (1-D): exact recall routing. The LLM decides; the
    # pipeline resolves deterministically.
    exact_recall_needed: bool = False
    exact_facets: list[ExactFacet] = Field(default_factory=list)
    answer_shape: AnswerShape = "open_domain"
    coverage_mode: CoverageMode = "top_support"
    source_precision: SourcePrecision = "preferred"

    @property
    def temporary_scaffolding(self) -> list[TemporaryScaffoldingTrace]:
        """Return trace-only temporary scaffolding labels."""
        return list(self._temporary_scaffolding)

    def model_copy_with_temporary_scaffolding(
        self,
        *,
        update: dict[str, Any] | None = None,
        additional: list[TemporaryScaffoldingTrace] | None = None,
    ) -> "QueryIntelligenceResult":
        """Copy while preserving trace-only scaffolding labels."""
        copied = self.model_copy(update=update)
        copied._temporary_scaffolding = _dedupe_temporary_scaffolding(
            [*self._temporary_scaffolding, *(additional or [])]
        )
        return copied

    @model_validator(mode="before")
    @classmethod
    def normalize_exact_recall_flag(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if not data.get("exact_facets"):
            return data
        normalized = dict(data)
        normalized["exact_recall_needed"] = True
        return normalized

    @field_validator("exact_facets")
    @classmethod
    def validate_exact_facets(cls, values: list[ExactFacet]) -> list[ExactFacet]:
        return _ensure_unique_values(values)

    @field_validator("sub_queries")
    @classmethod
    def validate_sub_queries(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values if value.strip()]
        if not normalized:
            raise ValueError("sub_queries must not be empty")
        return _ensure_unique_values(normalized)

    @field_validator("retrieval_levels")
    @classmethod
    def validate_retrieval_levels(cls, values: list[int]) -> list[int]:
        normalized = _ensure_unique_values(values)
        if not normalized:
            raise ValueError("retrieval_levels must not be empty")
        if any(value not in {0, 1, 2} for value in normalized):
            raise ValueError("retrieval_levels must contain only 0, 1, or 2")
        return normalized

    @field_validator("query_language", "answer_language", mode="before")
    @classmethod
    def validate_language_codes(cls, value: Any) -> str | None:
        return normalize_optional_iso_639_1_code(value)

    @model_validator(mode="after")
    def validate_sparse_query_hints_and_anchors(self) -> "QueryIntelligenceResult":
        (
            self.answer_shape,
            self.coverage_mode,
            self.source_precision,
        ) = derive_answer_coverage_fields(
            query_type=self.query_type,
            exact_recall_needed=self.exact_recall_needed,
            raw_context_access_mode=self.raw_context_access_mode,
        )
        sub_queries = set(self.sub_queries)
        if self.sparse_query_hints:
            seen_hint_targets: set[str] = set()
            normalized_hints: list[SparseQueryHint] = []
            for hint in self.sparse_query_hints:
                if hint.sub_query_text not in sub_queries:
                    raise ValueError("SparseQueryHint.sub_query_text must reference an item from sub_queries")
                if hint.sub_query_text in seen_hint_targets:
                    raise ValueError("SparseQueryHint.sub_query_text values must be unique")
                seen_hint_targets.add(hint.sub_query_text)
                normalized_hints.append(
                    _normalize_sparse_hint_precision_fields(
                        hint,
                        query_type=self.query_type,
                        callback_bias=self.callback_bias,
                    )
                )
            self.sparse_query_hints = normalized_hints
        seen_anchor_signatures: set[tuple[str, str, str]] = set()
        normalized_anchors: list[RuntimeAnchor] = []
        for anchor in self.anchors:
            if anchor.sub_query_text not in sub_queries:
                raise ValueError("RuntimeAnchor.sub_query_text must reference an item from sub_queries")
            signature = (
                anchor.sub_query_text,
                anchor.anchor_type,
                anchor.original_surface,
            )
            if signature in seen_anchor_signatures:
                continue
            seen_anchor_signatures.add(signature)
            normalized_anchors.append(anchor)
        self.anchors = normalized_anchors
        if self.callback_bias:
            for hint in self.sparse_query_hints:
                if hint.quoted_phrases or hint.must_keep_terms:
                    continue
                raise ValueError(
                    "callback sparse_query_hints must preserve an explicit anchor via quoted_phrases or must_keep_terms"
                )
        if self.query_type == "slot_fill":
            for hint in self.sparse_query_hints:
                if hint.quoted_phrases or hint.must_keep_terms:
                    continue
                raise ValueError(
                    "slot_fill sparse_query_hints must preserve concrete anchors via quoted_phrases or must_keep_terms"
                )
        if self.query_type == "broad_list":
            for hint in self.sparse_query_hints:
                if hint.quoted_phrases or hint.must_keep_terms:
                    continue
                raise ValueError(
                    "broad_list sparse_query_hints must preserve explicit facet anchors via quoted_phrases or must_keep_terms"
                )
        if self.query_type == "broad_list" and len(self.sparse_query_hints) > 1:
            signatures = [_sparse_hint_signature(hint) for hint in self.sparse_query_hints]
            if len(signatures) != len(set(signatures)):
                raise ValueError(
                    "broad_list sparse_query_hints must preserve distinct facet anchors across sub_queries"
                )
        return self


# QueryPlanCore is the lean primary need-detection schema. The detailed
# rationale is kept as a comment (not a docstring) so it does NOT inflate the
# json_schema the model receives on every interactive turn.
#
# It is the small "contract" the primary need-detection LLM call must satisfy.
# Relative to the rich QueryIntelligenceResult it drops the heavy part of the
# schema only: the structured `anchors` list with its nested
# `RuntimeAnchorAlias` arrays-of-objects (about half of the rich JSON schema
# and the source of the cross-field anchor validators). Anchors are produced
# by a conditional anchor review and merged back in server-side.
#
# Every other rich field is KEPT here because none is neutral when defaulted.
# `needs` drives retrieval boosts, `temporal_range` drives time filtering,
# `callback_bias` shapes hints, `retrieval_levels` selects evidence tiers, and
# `query_language` / `answer_language` are cheap scalars that feed the answer-
# language hint downstream (dropping them silently disables that feature for
# every turn). Keeping these scalars costs only a few hundred schema chars; the
# bloat lived entirely in the structured anchors.
#
# It performs per-field shape validation only. It does NOT raise on
# hint<->sub_query linkage problems; those are repaired mechanically by
# `need_detector_repair` before the rich object is built, so a medium model
# that drifts on cross-field linkage degrades gracefully instead of failing
# the whole structured call.
class QueryPlanCore(BaseModel):
    """Lean primary need-detection plan requested from the model."""

    model_config = ConfigDict(extra="ignore")

    needs: list[DetectedNeed] = Field(default_factory=list)
    temporal_range: TemporalQueryRange | None = None
    sub_queries: list[str] = Field(min_length=1, max_length=3)
    callback_bias: bool = False
    raw_context_access_mode: RawContextAccessMode = "normal"
    sparse_query_hints: list[SparseQueryHint] = Field(default_factory=list)
    query_language: str | None = None
    answer_language: str | None = None
    query_type: QueryType = "default"
    memory_needed: bool = True
    retrieval_levels: list[int] = Field(default_factory=lambda: [0])
    exact_recall_needed: bool = False
    exact_facets: list[ExactFacet] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_exact_recall_flag(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if not data.get("exact_facets"):
            return data
        normalized = dict(data)
        normalized["exact_recall_needed"] = True
        return normalized

    @field_validator("exact_facets")
    @classmethod
    def validate_exact_facets(cls, values: list[ExactFacet]) -> list[ExactFacet]:
        return _ensure_unique_values(values)

    @field_validator("sub_queries")
    @classmethod
    def validate_sub_queries(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values if value.strip()]
        if not normalized:
            raise ValueError("sub_queries must not be empty")
        return _ensure_unique_values(normalized)

    @field_validator("query_language", "answer_language", mode="before")
    @classmethod
    def validate_language_codes(cls, value: Any) -> str | None:
        return normalize_optional_iso_639_1_code(value)

    @field_validator("retrieval_levels")
    @classmethod
    def validate_retrieval_levels(cls, values: list[int]) -> list[int]:
        normalized = _ensure_unique_values(values)
        if not normalized:
            raise ValueError("retrieval_levels must not be empty")
        if any(value not in {0, 1, 2} for value in normalized):
            raise ValueError("retrieval_levels must contain only 0, 1, or 2")
        return normalized


class PlannedSubQuery(BaseModel):
    """Mechanical retrieval rewrites for one semantic sub-query."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    sparse_phrase: str | None = None
    quoted_phrases: list[str] = Field(default_factory=list)
    must_keep_terms: list[str] = Field(default_factory=list)
    fts_queries: list[str] = Field(min_length=1)
    fts_query_kinds: list[str] = Field(default_factory=list)

    @field_validator("text", "sparse_phrase")
    @classmethod
    def validate_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("text must be non-empty")
        return normalized

    @field_validator("quoted_phrases", "must_keep_terms")
    @classmethod
    def validate_sparse_lists(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values if value.strip()]
        return _ensure_unique_values(normalized)

    @field_validator("fts_queries")
    @classmethod
    def validate_fts_queries(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values if value.strip()]
        if not normalized:
            raise ValueError("fts_queries must not be empty")
        return _ensure_unique_values(normalized)

    @field_validator("fts_query_kinds")
    @classmethod
    def validate_fts_query_kinds(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values if value.strip()]
        return normalized

    @model_validator(mode="after")
    def validate_fts_query_kind_alignment(self) -> "PlannedSubQuery":
        if self.fts_query_kinds and len(self.fts_query_kinds) != len(self.fts_queries):
            raise ValueError("fts_query_kinds must align with fts_queries")
        return self


class RetrievalPlan(BaseModel):
    """Deterministic retrieval plan built before candidate search."""

    model_config = ConfigDict(extra="forbid")

    original_query: str | None = None
    assistant_mode_id: str
    workspace_id: str | None = None
    conversation_id: str
    user_persona_id: str | None = None
    platform_id: str = "default"
    character_id: str | None = None
    active_presence_id: str | None = None
    active_space_id: str | None = None
    active_space_boundary_mode: SpaceBoundaryMode | None = None
    active_mind_id: str | None = None
    mind_topology: MindTopology = MindTopology.UNIMIND
    active_embodiment_id: str | None = None
    cross_embodiment_mode: EmbodimentBoundaryMode = (
        EmbodimentBoundaryMode.DIRECT_IF_SAME_BODY
    )
    active_realm_id: str | None = None
    cross_realm_mode: CrossRealmMode = CrossRealmMode.NONE
    incognito: bool = False
    remember_across_chats: bool = True
    remember_across_devices: bool = True
    fts_queries: list[str] = Field(default_factory=list)
    sub_query_plans: list[PlannedSubQuery] = Field(default_factory=list)
    callback_bias: bool = False
    raw_context_access_mode: RawContextAccessMode = "normal"
    query_language: str | None = None
    answer_language: str | None = None
    query_type: QueryType = "default"
    scope_filter: list[MemoryScope] = Field(default_factory=list)
    status_filter: list[MemoryStatus] = Field(default_factory=list)
    vector_limit: int = Field(ge=0, default=0)
    max_candidates: int = Field(ge=0)
    max_context_items: int = Field(gt=0)
    privacy_ceiling: int = Field(ge=0, le=3)
    privacy_enforcement: Literal["enforce", "audit_only", "off"] = "enforce"
    allow_intimacy_context: bool = False
    allow_private_sensitivity: bool = False
    retrieval_levels: list[int] = Field(default_factory=lambda: [0])
    temporal_query_range: TemporalQueryRange | None = None
    consequence_search_enabled: bool = False
    require_evidence_regrounding: bool = False
    skip_retrieval: bool = False
    # Wave 1 batch 2 (1-D): exact recall lane. When enabled, downstream
    # stages prioritize raw evidence and concrete L0 memories over
    # abstracted summaries/beliefs.
    exact_recall_mode: bool = False
    exact_facets: list[ExactFacet] = Field(default_factory=list)
    answer_shape: AnswerShape = "open_domain"
    coverage_mode: CoverageMode = "top_support"
    source_precision: SourcePrecision = "preferred"

    @field_validator("fts_queries")
    @classmethod
    def validate_fts_queries(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values if value.strip()]
        return _ensure_unique_values(normalized)

    @field_validator("original_query")
    @classmethod
    def validate_original_query(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("query_language", "answer_language", mode="before")
    @classmethod
    def validate_plan_language_codes(cls, value: Any) -> str | None:
        return normalize_optional_iso_639_1_code(value)

    @field_validator("sub_query_plans")
    @classmethod
    def validate_sub_query_plans(cls, values: list[PlannedSubQuery]) -> list[PlannedSubQuery]:
        if not values:
            return values
        texts = [value.text for value in values]
        _ensure_unique_values(texts)
        return values

    @field_validator("scope_filter")
    @classmethod
    def validate_scope_filter(cls, values: list[MemoryScope]) -> list[MemoryScope]:
        return _ensure_unique_values(values)

    @field_validator("status_filter")
    @classmethod
    def validate_status_filter(cls, values: list[MemoryStatus]) -> list[MemoryStatus]:
        return _ensure_unique_values(values)

    @field_validator("retrieval_levels")
    @classmethod
    def validate_retrieval_levels(cls, values: list[int]) -> list[int]:
        normalized = _ensure_unique_values(values)
        if not normalized:
            raise ValueError("retrieval_levels must not be empty")
        if any(value not in {0, 1, 2} for value in normalized):
            raise ValueError("retrieval_levels must contain only 0, 1, or 2")
        return normalized

    @field_validator("exact_facets")
    @classmethod
    def validate_plan_exact_facets(cls, values: list[ExactFacet]) -> list[ExactFacet]:
        return _ensure_unique_values(values)

    @model_validator(mode="after")
    def normalize_answer_coverage_fields(self) -> "RetrievalPlan":
        (
            self.answer_shape,
            self.coverage_mode,
            self.source_precision,
        ) = derive_answer_coverage_fields(
            query_type=self.query_type,
            exact_recall_needed=self.exact_recall_mode,
            raw_context_access_mode=self.raw_context_access_mode,
        )
        return self


class ScoredCandidate(BaseModel):
    """Scored retrieval candidate ready for context composition."""

    model_config = ConfigDict(extra="forbid")

    memory_id: str
    memory_object: dict[str, Any]
    llm_applicability: float = Field(ge=0.0, le=1.0)
    retrieval_score: float = Field(ge=0.0, le=1.0)
    vitality_boost: float = Field(ge=0.0, le=1.0)
    confirmation_boost: float = Field(ge=0.0, le=1.0)
    need_boost: float = Field(ge=0.0)
    penalty: float = Field(ge=0.0)
    final_score: float
    resolved_date: str | None = None


class ComposedContext(BaseModel):
    """Ephemeral prompt-ready context assembled for a single response."""

    model_config = ConfigDict(extra="forbid")

    contract_block: str = ""
    workspace_block: str = ""
    answer_evidence_block: str = ""
    answer_evidence_memory_ids: list[str] = Field(default_factory=list)
    answer_evidence_items: list[dict[str, Any]] = Field(default_factory=list)
    answer_evidence_sufficiency: dict[str, Any] = Field(default_factory=dict)
    answer_shape: AnswerShape = "open_domain"
    coverage_mode: CoverageMode = "top_support"
    source_precision: SourcePrecision = "preferred"
    coverage_state: EvidenceCoverageState = "unknown"
    support_map: dict[str, list[str]] = Field(default_factory=dict)
    allowed_values: list[dict[str, Any]] = Field(default_factory=list)
    missing_slots: list[dict[str, Any]] = Field(default_factory=list)
    memory_block: str = ""
    state_block: str = ""
    selected_memory_ids: list[str] = Field(default_factory=list)
    total_tokens_estimate: int = Field(ge=0)
    budget_tokens: int = Field(ge=0)
    items_included: int = Field(ge=0)
    items_dropped: int = Field(ge=0)


class MemoryObject(BaseModel):
    """Canonical memory object."""

    model_config = ConfigDict(extra="forbid")

    id: str
    user_id: str
    workspace_id: str | None = None
    conversation_id: str | None = None
    assistant_mode_id: str | None = None
    # Namespace redesign identity axes. `user_persona_id`, `platform_id` and
    # `character_id` are explicit caller inputs (never inferred from prompt
    # text). `platform_id` is required at write time after the public cut;
    # storage keeps it nullable for legacy rows during the additive phase.
    user_persona_id: str | None = None
    platform_id: str | None = None
    character_id: str | None = None
    active_presence_id: str | None = None
    source_presence_id: str | None = None
    presence_cluster_id: str | None = None
    space_id: str | None = None
    space_boundary_mode: SpaceBoundaryMode | None = None
    memory_owner_id: str | None = None
    source_mind_id: str | None = None
    embodiment_id: str | None = None
    realm_id: str | None = None
    object_type: MemoryObjectType
    scope: MemoryScope
    canonical_text: str
    index_text: str | None = None
    payload_json: dict[str, Any] = Field(default_factory=dict)
    source_kind: MemorySourceKind
    confidence: float = 0.5
    stability: float = 0.5
    vitality: float = 0.0
    maya_score: float = 0.0
    privacy_level: int = 0
    sensitivity: MemorySensitivity = MemorySensitivity.UNKNOWN
    themes: list[str] = Field(default_factory=list)
    auto_expires: bool = False
    platform_locked: bool = False
    platform_id_lock: str | None = None
    memory_category: MemoryCategory = MemoryCategory.UNKNOWN
    intimacy_boundary: IntimacyBoundary = IntimacyBoundary.ORDINARY
    intimacy_boundary_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    preserve_verbatim: bool = False
    temporal_type: str = "unknown"
    tension_score: float = 0.0
    tension_updated_at: datetime | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    status: MemoryStatus = MemoryStatus.ACTIVE
    created_at: datetime
    updated_at: datetime


class ContractDimensionCurrent(BaseModel):
    """Projection row for an interaction contract dimension."""

    model_config = ConfigDict(extra="forbid")

    id: str
    user_id: str
    workspace_id: str | None = None
    conversation_id: str | None = None
    assistant_mode_id: str | None = None
    user_persona_id: str | None = None
    platform_id: str | None = None
    character_id: str | None = None
    scope: MemoryScope
    dimension_name: str
    value_json: dict[str, Any] = Field(default_factory=dict)
    confidence: float
    source_memory_id: str
    updated_at: datetime
    sensitivity: MemorySensitivity = MemorySensitivity.UNKNOWN
    themes: list[str] = Field(default_factory=list)
    platform_locked: bool = False
    platform_id_lock: str | None = None


class MemoryConsentProfile(BaseModel):
    """Per-user consent counts for sensitive memory categories.

    The plan re-keys consent to `(user_id, user_persona_key, category)` so
    sensitive interactions in one persona never affect another. The base
    identity is represented by `user_persona_id=None`.
    """

    model_config = ConfigDict(extra="forbid")

    user_id: str
    user_persona_id: str | None = None
    category: MemoryCategory
    confirmed_count: int = Field(ge=0)
    declined_count: int = Field(ge=0)
    last_confirmed_at: datetime | None = None
    last_declined_at: datetime | None = None
    updated_at: datetime


class RetrievalEvent(BaseModel):
    """Trace of a retrieval operation."""

    model_config = ConfigDict(extra="forbid")

    id: str
    user_id: str
    conversation_id: str
    request_message_id: str
    response_message_id: str | None = None
    assistant_mode_id: str
    retrieval_plan_json: dict[str, Any]
    selected_memory_ids_json: list[str] = Field(default_factory=list)
    context_view_json: dict[str, Any] = Field(default_factory=dict)
    outcome_json: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


# ---------------------------------------------------------------------------
# Retrieval pipeline tracing (Wave 0-B instrumentation)
# ---------------------------------------------------------------------------


class RuntimeAliasSurfaceTrace(BaseModel):
    """Content-minimal runtime alias diagnostic; never evidential proof."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    surface: str
    alias_kind: AliasKind
    alias_language: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    non_evidential: Literal[True] = True

    @field_validator("surface")
    @classmethod
    def validate_surface(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("surface must be non-empty")
        return normalized

    @field_validator("alias_language")
    @classmethod
    def validate_alias_language(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class RuntimeAliasGroupTrace(BaseModel):
    """Runtime aliases grouped by their source sub-query and anchor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sub_query_text: str
    anchor_type: AnchorType
    original_surface: str
    normalized_surface: str | None = None
    preserve_verbatim: bool = False
    anchor_confidence: float = Field(ge=0.0, le=1.0)
    anchor_non_evidential: Literal[True] = True
    aliases: list[RuntimeAliasSurfaceTrace] = Field(default_factory=list)

    @field_validator("sub_query_text", "original_surface")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("runtime alias group required text must be non-empty")
        return normalized

    @field_validator("normalized_surface")
    @classmethod
    def validate_normalized_surface(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


def _normalize_profile_language_code(value: str) -> str:
    return normalize_iso_639_1_code(value)


def _normalize_profile_text(value: str, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


class LanguageProfileSourceRef(BaseModel):
    """Source reference proving one user communication profile row."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_kind: Literal["source_message", "message_window", "memory_object"]
    memory_id: str | None = None
    conversation_id: str | None = None
    source_message_id: str | None = None
    from_seq: int | None = Field(default=None, ge=0)
    to_seq: int | None = Field(default=None, ge=0)

    @field_validator("memory_id", "conversation_id", "source_message_id")
    @classmethod
    def validate_optional_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @model_validator(mode="after")
    def validate_source_reference(self) -> "LanguageProfileSourceRef":
        if self.source_kind == "memory_object" and not self.memory_id:
            raise ValueError("memory_object source refs require memory_id")
        if self.source_kind == "source_message" and not self.source_message_id:
            raise ValueError("source_message source refs require source_message_id")
        if self.source_kind == "message_window":
            if not self.conversation_id:
                raise ValueError("message_window source refs require conversation_id")
            if self.from_seq is None or self.to_seq is None:
                raise ValueError("message_window source refs require from_seq and to_seq")
            if self.from_seq > self.to_seq:
                raise ValueError("message_window from_seq must be <= to_seq")
        if not (self.memory_id or self.conversation_id or self.source_message_id):
            raise ValueError("source refs require at least one concrete source id")
        return self


class ObservedUserLanguage(BaseModel):
    """Observed user-authored language use; not a fluency claim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    language_code: str
    evidence_kind: Literal["user_authored_messages"] = "user_authored_messages"
    message_count: int = Field(ge=1)
    last_seen_at: str | None = None
    context_label: str = "default"
    source_refs: list[LanguageProfileSourceRef] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("language_code")
    @classmethod
    def validate_language_code(cls, value: str) -> str:
        return _normalize_profile_language_code(value)

    @field_validator("context_label")
    @classmethod
    def validate_context_label(cls, value: str) -> str:
        return _normalize_profile_text(value, field_name="context_label")

    @field_validator("last_seen_at")
    @classmethod
    def validate_last_seen_at(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class ExplicitLanguagePreference(BaseModel):
    """A source-backed language preference stated by the user."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    language_code: str
    preference_kind: Literal[
        "default_answer_language",
        "contextual_answer_language",
        "avoid_language",
        "terms_or_code_language",
    ]
    context_label: str = "default"
    source_refs: list[LanguageProfileSourceRef] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("language_code")
    @classmethod
    def validate_language_code(cls, value: str) -> str:
        return _normalize_profile_language_code(value)

    @field_validator("context_label")
    @classmethod
    def validate_context_label(cls, value: str) -> str:
        return _normalize_profile_text(value, field_name="context_label")


class ExplicitLanguageAbility(BaseModel):
    """A source-backed claim about language ability, separate from observation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    language_code: str
    ability_kind: Literal["speaks", "understands", "native", "fluent", "learning"]
    source_refs: list[LanguageProfileSourceRef] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("language_code")
    @classmethod
    def validate_language_code(cls, value: str) -> str:
        return _normalize_profile_language_code(value)


class LanguageContextualNorm(BaseModel):
    """A source-backed norm for language behavior in one context."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    language_code: str
    norm_kind: Literal[
        "default_answer_language",
        "comfortable_for_terms_or_code",
        "work_language",
        "personal_language",
        "language_switch_ok",
    ]
    context_label: str
    source_refs: list[LanguageProfileSourceRef] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("language_code")
    @classmethod
    def validate_language_code(cls, value: str) -> str:
        return _normalize_profile_language_code(value)

    @field_validator("context_label")
    @classmethod
    def validate_context_label(cls, value: str) -> str:
        return _normalize_profile_text(value, field_name="context_label")


class UserCommunicationProfile(BaseModel):
    """Derived control-plane memory of how the user communicates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_kind: Literal["user_language_profile"] = "user_language_profile"
    profile_version: Literal[1] = 1
    subject_presence_id: str | None = None
    observed_user_languages: list[ObservedUserLanguage] = Field(default_factory=list)
    explicit_language_preferences: list[ExplicitLanguagePreference] = Field(
        default_factory=list
    )
    explicit_language_abilities: list[ExplicitLanguageAbility] = Field(
        default_factory=list
    )
    contextual_norms: list[LanguageContextualNorm] = Field(default_factory=list)
    stale: bool = False
    stale_reason: str | None = None
    external_content_languages_excluded: Literal[True] = True
    control_plane_only: Literal[True] = True

    @field_validator("subject_presence_id", "stale_reason")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @model_validator(mode="after")
    def validate_stale_reason(self) -> "UserCommunicationProfile":
        if self.stale and not self.stale_reason:
            raise ValueError("stale profiles require stale_reason")
        return self


class UserCommunicationProfileTrace(BaseModel):
    """Content-minimal trace of user communication profile use."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_kind: Literal["user_language_profile"] = "user_language_profile"
    profile_version: int = Field(ge=1)
    stale: bool = False
    observed_language_codes: list[str] = Field(default_factory=list)
    preference_language_codes: list[str] = Field(default_factory=list)
    ability_language_codes: list[str] = Field(default_factory=list)
    contextual_norm_language_codes: list[str] = Field(default_factory=list)
    control_plane_only: Literal[True] = True

    @field_validator(
        "observed_language_codes",
        "preference_language_codes",
        "ability_language_codes",
        "contextual_norm_language_codes",
    )
    @classmethod
    def validate_language_codes(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            code = _normalize_profile_language_code(value)
            if code in seen:
                continue
            seen.add(code)
            normalized.append(code)
        return normalized


class ContentLanguageProfileTraceRow(BaseModel):
    """Content-free language metadata row used by need-detection diagnostics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    language_code: str
    memory_count: int = Field(ge=0)
    last_seen_at: str | None = None

    @field_validator("language_code")
    @classmethod
    def validate_language_code(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("language_code must be non-empty")
        return normalized

    @field_validator("last_seen_at")
    @classmethod
    def validate_last_seen_at(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class NeedDetectionTrace(BaseModel):
    """Trace of the need detection stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    detected_needs: list[str] = Field(default_factory=list)
    sub_queries: list[str] = Field(default_factory=list)
    sparse_hints: list[str] = Field(default_factory=list)
    query_language: str | None = None
    answer_language: str | None = None
    content_language_profile: list[ContentLanguageProfileTraceRow] = Field(
        default_factory=list
    )
    user_communication_profile: UserCommunicationProfileTrace | None = None
    anchors: list[RuntimeAnchor] = Field(default_factory=list)
    alias_groups: list[RuntimeAliasGroupTrace] = Field(default_factory=list)
    query_type: str = "default"
    raw_context_access_mode: str = "normal"
    answer_shape: AnswerShape = "open_domain"
    coverage_mode: CoverageMode = "top_support"
    source_precision: SourcePrecision = "preferred"
    temporal_range: str | None = None
    retrieval_levels: list[int] = Field(default_factory=lambda: [0])
    degraded_mode: bool = False
    exact_recall_needed: bool = False
    exact_facets: list[str] = Field(default_factory=list)
    temporary_scaffolding: list[TemporaryScaffoldingTrace] = Field(default_factory=list)
    duration_ms: float = Field(ge=0.0)


class FtsQueryExecutionCount(BaseModel):
    """Per-FTS-query execution diagnostics for one sub-query."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str
    kind: str = "unknown"
    match_mode: str = "implicit_and"
    source: str = "planned"
    non_evidential: Literal[True] = True
    # Rows returned by SQL after filters and channel limit, before merge/fusion.
    raw_rows: int = Field(ge=0, default=0)
    candidates: int = Field(ge=0, default=0)


class SubQuerySearchCount(BaseModel):
    """Per-sub-query candidate counts from search."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    subquery: str
    verbatim_pin: int = Field(ge=0, default=0)
    artifact_chunk: int = Field(ge=0, default=0)
    fts: int = Field(ge=0, default=0)
    fact_facet: int = Field(ge=0, default=0)
    embedding: int = Field(ge=0, default=0)
    verbatim_evidence_search: int = Field(ge=0, default=0)
    fts_queries: list[str] = Field(default_factory=list)
    fts_query_kinds: list[str] = Field(default_factory=list)
    fts_query_executions: list[FtsQueryExecutionCount] = Field(default_factory=list)


class CandidateSearchTrace(BaseModel):
    """Trace of the candidate search stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fts_candidates_count: int = Field(ge=0, default=0)
    verbatim_pin_candidates_count: int = Field(ge=0, default=0)
    artifact_chunk_candidates_count: int = Field(ge=0, default=0)
    fact_facet_candidates_count: int = Field(ge=0, default=0)
    embedding_candidates_count: int = Field(ge=0, default=0)
    consequence_candidates_count: int = Field(ge=0, default=0)
    verbatim_evidence_search_candidates_count: int = Field(ge=0, default=0)
    # Stubbed until the entity channel lands (Wave 2 / Phase 4). Reads
    # as 0 in every trace today; do not interpret it as "entity channel
    # ran and found nothing".
    entity_candidates_count: int = Field(ge=0, default=0)
    total_before_fusion: int = Field(ge=0, default=0)
    total_after_fusion: int = Field(ge=0, default=0)
    per_subquery_counts: list[SubQuerySearchCount] = Field(default_factory=list)
    duration_ms: float = Field(ge=0.0)


class ScoringTrace(BaseModel):
    """Trace of the applicability scoring stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidates_received: int = Field(ge=0, default=0)
    candidates_scored: int = Field(ge=0, default=0)
    candidates_rejected: int = Field(ge=0, default=0)
    rejection_reasons: dict[str, int] = Field(default_factory=dict)
    policy_audit_reason_counts: dict[str, int] = Field(default_factory=dict)
    top_score: float = 0.0
    median_score: float = 0.0
    min_score: float = 0.0
    applicability_gate_mode: Literal["off", "shadow", "enforced"] = "off"
    eligible_candidate_count: int = Field(ge=0, default=0)
    ineligible_reason_counts: dict[str, int] = Field(default_factory=dict)
    llm_applicability_skipped_count: int = Field(ge=0, default=0)
    shadow_disagreement_count: int = Field(ge=0, default=0)
    shadow_harmful_disagreement_count: int = Field(ge=0, default=0)
    estimated_calls_saved: int = Field(ge=0, default=0)
    adjacent_rrf_delta_distribution: dict[str, float] = Field(default_factory=dict)
    gate_reason: str = "mode_off"
    duration_ms: float = Field(ge=0.0)


class CompositionTrace(BaseModel):
    """Trace of the context composition stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidates_selected: int = Field(ge=0, default=0)
    token_budget_total: int = Field(ge=0, default=0)
    token_budget_used: int = Field(ge=0, default=0)
    contract_tokens: int = Field(ge=0, default=0)
    workspace_tokens: int = Field(ge=0, default=0)
    memory_tokens: int = Field(ge=0, default=0)
    state_tokens: int = Field(ge=0, default=0)
    diversity_penalties_applied: int = Field(ge=0, default=0)
    support_level: str = "UNKNOWN"
    answer_shape: AnswerShape = "open_domain"
    coverage_mode: CoverageMode = "top_support"
    source_precision: SourcePrecision = "preferred"
    coverage_state: EvidenceCoverageState = "unknown"
    duration_ms: float = Field(ge=0.0)


RetrievalSufficiencyState = Literal[
    "retrieval_sufficient",
    "retrieval_insufficient",
    "insufficient_no_candidates",
    "insufficient_no_scored_candidates",
    "insufficient_need_more_raw_evidence",
    "insufficient_need_artifact",
    "insufficient_summary_support",
    "contradictory_candidates",
]

RetrievalExpansionChannel = Literal[
    "fts",
    "embedding",
    "verbatim_evidence_search",
    "artifact_chunk",
    "consequence",
    "verbatim_pin",
]

RetrievalSufficiencyRationaleCode = Literal[
    "raw_candidates_empty",
    "scored_candidates_empty",
    "artifact_requested_no_artifact_candidates",
    "raw_evidence_requested_no_direct_candidates",
    "unsupported_summary_only",
    "top_candidate_unsupported_summary",
    "contradictory_belief_candidate",
    "top_score_below_floor",
    "scored_candidates_available",
]


class RetrievalSufficiencyDiagnostic(BaseModel):
    """Text-free shadow diagnostic for retrieval sufficiency."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    state: RetrievalSufficiencyState
    confidence: float = Field(ge=0.0, le=1.0)
    rationale_codes: list[RetrievalSufficiencyRationaleCode] = Field(default_factory=list)
    would_expand_channels: list[RetrievalExpansionChannel] = Field(default_factory=list)
    would_abstain: bool = False
    candidate_count: int = Field(ge=0, default=0)
    filtered_candidate_count: int = Field(ge=0, default=0)
    shortlist_count: int = Field(ge=0, default=0)
    scored_candidate_count: int = Field(ge=0, default=0)
    top_score: float = Field(ge=0.0, default=0.0)
    direct_evidence_candidate_count: int = Field(ge=0, default=0)
    summary_candidate_count: int = Field(ge=0, default=0)
    verbatim_evidence_search_candidate_count: int = Field(ge=0, default=0)
    artifact_candidate_count: int = Field(ge=0, default=0)
    unsupported_summary_candidate_count: int = Field(ge=0, default=0)
    contradictory_candidate_count: int = Field(ge=0, default=0)


FacetObligationStatus = Literal["covered", "partial", "missing"]
FacetSupportVerdict = Literal["supported", "unsupported"]


class FacetSupportObligationTrace(BaseModel):
    """Text-free support trace for one retrieval obligation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    description: str
    status: FacetObligationStatus
    selected_memory_ids: list[str] = Field(default_factory=list)
    composed_memory_ids: list[str] = Field(default_factory=list)
    support_verdict: FacetSupportVerdict = "unsupported"


class FacetSupportTrace(BaseModel):
    """Facet/obligation coverage diagnostics for stability gates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    obligations: list[FacetSupportObligationTrace] = Field(default_factory=list)


ProofSourceKind = Literal[
    "base_canonical",
    "summary_joined",
    "summary_only",
    "raw_source_span",
    "raw_only",
    "derived_only",
]


class ProvenanceEvidenceTrace(BaseModel):
    """Selected evidence provenance without raw memory text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    memory_id: str
    recovery_channels: list[str] = Field(default_factory=list)
    proof_source: ProofSourceKind
    joined_to_base: bool = False
    selected: bool = False
    matched_subquery_indexes: list[int] = Field(default_factory=list)
    scope: str | None = None
    scope_canonical: str | None = None
    conversation_id: str | None = None


class DirectVsIndirectProvenanceTrace(BaseModel):
    """Direct/indirect recovery provenance for selected context."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence: list[ProvenanceEvidenceTrace] = Field(default_factory=list)
    direct_recovery_count: int = Field(ge=0, default=0)
    indirect_recovery_count: int = Field(ge=0, default=0)
    summary_only_count: int = Field(ge=0, default=0)
    raw_cross_conversation_count: int = Field(ge=0, default=0)


class TokenBudgetTrace(BaseModel):
    """Retrieval-time token budget diagnostics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_tokens: int | None = Field(default=None, ge=0)
    context_tokens: int = Field(ge=0, default=0)
    answer_max_tokens: int | None = Field(default=None, ge=0)
    finish_reason: str | None = None
    postcondition_retry_count: int = Field(ge=0, default=0)
    output_limit_seen: bool = False
    context_compression_used: bool = False
    token_budget_total: int = Field(ge=0, default=0)
    token_budget_used: int = Field(ge=0, default=0)
    contract_tokens: int = Field(ge=0, default=0)
    workspace_tokens: int = Field(ge=0, default=0)
    memory_tokens: int = Field(ge=0, default=0)
    state_tokens: int = Field(ge=0, default=0)


class CrossConversationRawPolicyTrace(BaseModel):
    """Policy trace for cross-conversation raw evidence recovery."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    activation_reason: str | None = None
    candidate_count: int = Field(ge=0, default=0)
    selected_count: int = Field(ge=0, default=0)
    violation_count: int = Field(ge=0, default=0)
    max_windows_per_subquery: int = Field(ge=0, default=0)
    policy_filters: list[str] = Field(default_factory=list)


class RetrievalCustodyTrace(BaseModel):
    """Stage-level retrieval custody counts without raw candidate text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_candidate_count: int = Field(ge=0, default=0)
    candidate_count_by_channel: dict[str, int] = Field(default_factory=dict)
    source_backed_candidate_count: int = Field(ge=0, default=0)
    summary_only_candidate_count: int = Field(ge=0, default=0)
    post_user_id_candidate_count: int = Field(ge=0, default=0)
    post_scope_coordinate_lifecycle_candidate_count: int = Field(ge=0, default=0)
    scored_candidate_count: int = Field(ge=0, default=0)
    selected_candidate_count: int = Field(ge=0, default=0)
    selected_evidence_ids: list[str] = Field(default_factory=list)
    selected_source_evidence_count: int = Field(ge=0, default=0)
    selected_summary_count: int = Field(ge=0, default=0)
    high_value_rejected_candidate_count: int = Field(ge=0, default=0)
    high_value_rejected_reasons: dict[str, int] = Field(default_factory=dict)
    candidate_found_but_not_selected: list[str] = Field(default_factory=list)
    rendered_evidence_ids: list[str] = Field(default_factory=list)
    funnel_coverage_state: EvidenceCoverageState = "unknown"
    source_window_ids: list[str] = Field(default_factory=list)
    selected_source_window_ids: list[str] = Field(default_factory=list)
    drop_counts_by_stage: dict[str, int] = Field(default_factory=dict)
    drop_counts_by_reason: dict[str, int] = Field(default_factory=dict)


class RequestRuntimeDiagnosticsTrace(BaseModel):
    """Request-path cost and latency counters for retrieval diagnostics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage_timings_ms: dict[str, float] = Field(default_factory=dict)
    db_query_count: int = Field(ge=0, default=0)
    db_query_count_by_operation: dict[str, int] = Field(default_factory=dict)
    hydration_timings_ms: dict[str, float] = Field(default_factory=dict)
    lock_wait_count: int = Field(ge=0, default=0)
    sqlite_busy_count: int = Field(ge=0, default=0)


class TopicTraceItem(BaseModel):
    """Compact read-only topic item included in retrieval traces."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    status: str
    title: str
    summary: str
    active_goal: str | None = None
    open_questions: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    source_counts: dict[str, int] = Field(default_factory=dict)
    source_refs: list[dict[str, str]] = Field(default_factory=list)
    source_message_start_seq: int | None = None
    source_message_end_seq: int | None = None
    last_touched_seq: int | None = None
    last_touched_at: str | None = None
    confidence: float | None = None
    privacy_level: int | None = None
    sensitivity: MemorySensitivity = MemorySensitivity.PUBLIC
    themes: list[str] = Field(default_factory=list)
    platform_locked: bool = False
    platform_id_lock: str | None = None
    intimacy_boundary: IntimacyBoundary = IntimacyBoundary.ORDINARY
    intimacy_boundary_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class TopicWorkingSetFreshness(BaseModel):
    """Freshness metadata for the conversation topic orientation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str = "fresh"
    last_processed_seq: int | None = None
    last_processed_message_id: str | None = None
    latest_message_seq: int = Field(ge=0, default=0)
    lag_message_count: int = Field(ge=0, default=0)
    lag_token_count: int = Field(ge=0, default=0)
    refresh_message_threshold: int = Field(ge=1, default=4)
    stale_message_threshold: int = Field(ge=1, default=10)
    refresh_token_threshold: int = Field(ge=1, default=2000)
    stale_token_threshold: int = Field(ge=1, default=5000)


class TopicWorkingSetTrace(BaseModel):
    """Read-only snapshot of active and parked conversation topics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    active_topics: list[TopicTraceItem] = Field(default_factory=list)
    parked_topics: list[TopicTraceItem] = Field(default_factory=list)
    freshness: TopicWorkingSetFreshness = Field(default_factory=TopicWorkingSetFreshness)


StructuredOutputDiagnosticEvent = Literal[
    "invalid_structured_output",
    "malformed_domain_output",
    "missing_after_retry",
    "output_limit_drop",
    "output_limit_split",
]


class LLMStructuredOutputDiagnostic(BaseModel):
    """Structured-output issue observed during retrieval-time LLM calls."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event: StructuredOutputDiagnosticEvent
    purpose: str
    model: str | None = None
    candidate_count: int = Field(ge=0, default=0)
    returned_count: int = Field(ge=0, default=0)
    accepted_count: int = Field(ge=0, default=0)
    malformed_count: int = Field(ge=0, default=0)
    unknown_count: int = Field(ge=0, default=0)
    duplicate_count: int = Field(ge=0, default=0)
    missing_count: int = Field(ge=0, default=0)
    retry_count: int = Field(ge=0, default=0)
    details: dict[str, Any] = Field(default_factory=dict)
    debug_artifact_path: str | None = None


class RetrievalTrace(BaseModel):
    """Complete trace of a retrieval operation."""

    model_config = ConfigDict(extra="forbid")

    query_text: str
    user_id: str
    conversation_id: str
    requested_mode: str | None = None
    effective_mode: str | None = None
    response_mode: Literal["normal", "fast", "smart_fast"] = "normal"
    timestamp_iso: str
    small_corpus_mode: bool = False
    degraded_mode: bool = False
    raw_context_access_mode: str = "normal"
    privacy_enforcement: Literal["enforce", "audit_only", "off"] = "enforce"
    policy_filter_audit: dict[str, Any] = Field(default_factory=dict)
    topic_snapshot: TopicWorkingSetTrace = Field(default_factory=TopicWorkingSetTrace)
    need_detection: NeedDetectionTrace | None = None
    candidate_search: CandidateSearchTrace | None = None
    scoring: ScoringTrace | None = None
    composition: CompositionTrace | None = None
    retrieval_sufficiency: RetrievalSufficiencyDiagnostic | None = None
    facet_support: FacetSupportTrace | None = None
    direct_vs_indirect_provenance: DirectVsIndirectProvenanceTrace | None = None
    token_budget: TokenBudgetTrace | None = None
    cross_conversation_raw_policy: CrossConversationRawPolicyTrace | None = None
    custody: RetrievalCustodyTrace = Field(default_factory=RetrievalCustodyTrace)
    runtime_diagnostics: RequestRuntimeDiagnosticsTrace = Field(
        default_factory=RequestRuntimeDiagnosticsTrace
    )
    structured_output_diagnostics: list[LLMStructuredOutputDiagnostic] = Field(default_factory=list)
    temporary_scaffolding: list[TemporaryScaffoldingTrace] = Field(default_factory=list)
    total_duration_ms: float = Field(ge=0.0, default=0.0)


def _dedupe_temporary_scaffolding(
    events: list[TemporaryScaffoldingTrace],
) -> list[TemporaryScaffoldingTrace]:
    deduped: list[TemporaryScaffoldingTrace] = []
    seen: set[tuple[str, str, str]] = set()
    for event in events:
        key = (event.component, event.mechanism, event.trace_flag)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(event)
    return deduped
