"""Memory domain schemas."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
import json
import re
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class MemoryObjectType(str, Enum):
    EVIDENCE = "evidence"
    BELIEF = "belief"
    INTERACTION_CONTRACT = "interaction_contract"
    STATE_SNAPSHOT = "state_snapshot"
    CONSEQUENCE_CHAIN = "consequence_chain"
    SUMMARY_VIEW = "summary_view"


class MemoryScope(str, Enum):
    GLOBAL_USER = "global_user"
    ASSISTANT_MODE = "assistant_mode"
    WORKSPACE = "workspace"
    CONVERSATION = "conversation"
    EPHEMERAL_SESSION = "ephemeral_session"


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


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"
    DELETED = "deleted"
    REVIEW_REQUIRED = "review_required"
    PENDING_USER_CONFIRMATION = "pending_user_confirmation"
    DECLINED = "declined"


class SummaryViewKind(str, Enum):
    CONVERSATION_CHUNK = "conversation_chunk"
    WORKSPACE_ROLLUP = "workspace_rollup"
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


class AssistantModeId(str, Enum):
    CODING_DEBUG = "coding_debug"
    RESEARCH_DEEP_DIVE = "research_deep_dive"
    COMPANION = "companion"
    BRAINSTORM = "brainstorm"
    BIOGRAPHICAL_INTERVIEW = "biographical_interview"
    GENERAL_QA = "general_qa"
    PERSONAL_ASSISTANT = "personal_assistant"


TemporalType = Literal["permanent", "bounded", "event_triggered", "ephemeral", "unknown"]
QueryType = Literal["broad_list", "temporal", "slot_fill", "default"]
RawContextAccessMode = Literal["normal", "skipped_raw", "artifact", "verbatim"]

_VALID_TEMPORAL_TYPES = frozenset(get_args(TemporalType))
_QUERY_HINT_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
_FTS5_OPERATOR_TOKENS = frozenset({"and", "or", "not", "near"})


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
    """Adaptive context-cache settings attached to an assistant mode."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_ttl_seconds: int = Field(gt=0)
    sync_threshold: float = Field(gt=0.0, le=1.0)
    max_messages_without_refresh: int = Field(ge=1)
    max_minutes_without_refresh: int = Field(ge=1)


class AssistantModeManifest(BaseModel):
    """Validated assistant mode manifest loaded from JSON files."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    assistant_mode_id: AssistantModeId
    display_name: str = Field(min_length=1)
    cross_chat_allowed: bool
    allowed_scopes: list[MemoryScope] = Field(min_length=1)
    preferred_memory_types: list[MemoryObjectType] = Field(min_length=1)
    need_triggers: list[NeedTrigger] = Field(default_factory=list)
    contract_dimensions_priority: list[str] = Field(min_length=1)
    privacy_ceiling: int = Field(ge=0, le=3)
    context_budget_tokens: int = Field(gt=0)
    transcript_budget_tokens: int = Field(gt=0)
    retrieval_params: RetrievalParams
    context_cache_policy: ContextCachePolicy
    prompt_hash: str | None = Field(default=None, exclude=True)

    @field_validator("allowed_scopes")
    @classmethod
    def validate_allowed_scopes(cls, values: list[MemoryScope]) -> list[MemoryScope]:
        return _ensure_unique_values(values)

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

    role: str
    content: str


class ExtractionConversationContext(BaseModel):
    """Execution context for a single extraction run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str
    conversation_id: str
    source_message_id: str
    workspace_id: str | None = None
    assistant_mode_id: str
    recent_messages: list[ExtractionContextMessage] = Field(default_factory=list)


class ExtractedMemoryBase(BaseModel):
    """Shared fields returned by the extraction model."""

    model_config = ConfigDict(extra="ignore")

    canonical_text: str = Field(min_length=1)
    index_text: str | None = None
    scope: MemoryScope
    confidence: float = Field(ge=0.0, le=1.0)
    source_kind: MemorySourceKind
    privacy_level: int = Field(ge=0, le=3)
    memory_category: MemoryCategory = MemoryCategory.UNKNOWN
    preserve_verbatim: bool = False
    informational_mention: bool | None = None
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
            for field_name in ("text", "memory_text", "summary_text", "description", "content"):
                value = normalized.get(field_name)
                if isinstance(value, str) and value.strip():
                    normalized["canonical_text"] = value
                    break
        if "confidence" not in normalized and "score" in normalized:
            normalized["confidence"] = normalized.get("score")
        normalized.setdefault("confidence", 0.5)
        normalized.setdefault("scope", MemoryScope.CONVERSATION.value)
        normalized.setdefault("source_kind", MemorySourceKind.EXTRACTED.value)
        normalized.setdefault("privacy_level", 0)
        normalized.setdefault("payload", {})
        normalized.setdefault("temporal_type", "unknown")
        normalized.setdefault("temporal_confidence", 0.0)
        return normalized

    @field_validator("index_text")
    @classmethod
    def validate_index_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

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
                "language_codes must contain at least one ISO 639-1 code "
                "for the language of canonical_text"
            )
        seen: set[str] = set()
        for code in self.language_codes:
            if not isinstance(code, str) or len(code) != 2:
                raise ValueError(f"invalid ISO 639-1 code: {code!r}")
            lowered = code.lower()
            if not lowered.isalpha() or not lowered.isascii():
                raise ValueError(f"invalid ISO 639-1 code: {code!r}")
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
            if isinstance(item, dict) and (
                item.get("claim_key") is None or item.get("claim_value") is None
            ):
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
                item_type = str(
                    normalized_item.get("memory_type")
                    or normalized_item.get("item_type")
                    or normalized_item.get("object_type")
                    or normalized_item.get("type")
                    or normalized_item.get("kind")
                    or ""
                ).strip().lower()
                if (
                    ("claim_key" in normalized_item and "claim_value" in normalized_item)
                    or (
                        item_type == MemoryObjectType.BELIEF.value
                        and normalized_item.get("claim_key") is not None
                        and normalized_item.get("claim_value") is not None
                    )
                ):
                    claim_key = str(normalized_item.get("claim_key") or "").strip()
                    claim_value_raw = normalized_item.get("claim_value")
                    if isinstance(claim_value_raw, str):
                        claim_value = claim_value_raw.strip()
                    elif claim_value_raw is None:
                        claim_value = ""
                    else:
                        claim_value = json.dumps(claim_value_raw, ensure_ascii=False, sort_keys=True)
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
            for field_name in ("evidences", "beliefs", "contract_signals", "state_updates"):
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
        return normalized

    @model_validator(mode="after")
    def validate_nothing_durable_consistency(self) -> "ExtractionResult":
        if self.nothing_durable and (
            self.evidences or self.beliefs or self.contract_signals or self.state_updates
        ):
            raise ValueError("nothing_durable=true but extraction lists are non-empty")
        return self


class ContractSignal(BaseModel):
    """Contract preference signal extracted from a message."""

    model_config = ConfigDict(extra="forbid")

    canonical_text: str = Field(min_length=1)
    dimension_name: str = Field(min_length=1)
    value_json: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    scope: MemoryScope
    source_kind: MemorySourceKind = MemorySourceKind.INFERRED
    privacy_level: int = Field(ge=0, le=3, default=1)

    @field_validator("dimension_name")
    @classmethod
    def validate_dimension_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("dimension_name must be non-empty")
        return normalized


class ContractProjectionResult(BaseModel):
    """Structured output returned by the contract projector."""

    model_config = ConfigDict(extra="forbid")

    signals: list[ContractSignal] = Field(default_factory=list)
    nothing_durable: bool = False

    @model_validator(mode="after")
    def validate_nothing_durable_consistency(self) -> "ContractProjectionResult":
        if self.nothing_durable and self.signals:
            raise ValueError("nothing_durable=true but contract signals are non-empty")
        return self


class DetectedNeed(BaseModel):
    """Need signal detected from the active message and recent context."""

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

    is_consequence: bool
    action_description: str = ""
    outcome_description: str = ""
    outcome_sentiment: ConsequenceSentiment = ConsequenceSentiment.NEUTRAL
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    likely_action_message_id: str | None = None

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

    model_config = ConfigDict(extra="forbid")

    start: datetime
    end: datetime

    @model_validator(mode="after")
    def validate_bounds(self) -> "TemporalQueryRange":
        if self.end < self.start:
            raise ValueError("TemporalQueryRange.end must be >= start")
        return self


class SparseQueryHint(BaseModel):
    """Sparse lexical shaping hints for one semantic sub-query."""

    model_config = ConfigDict(extra="forbid")

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
            raise ValueError(
                "SparseQueryHint requires at least one of fts_phrase, quoted_phrases, or must_keep_terms"
            )
        return self


class QueryIntelligenceResult(BaseModel):
    """Structured query understanding returned by need detection."""

    model_config = ConfigDict(extra="forbid")

    needs: list[DetectedNeed] = Field(default_factory=list)
    temporal_range: TemporalQueryRange | None = None
    sub_queries: list[str] = Field(min_length=1, max_length=3)
    callback_bias: bool = False
    raw_context_access_mode: RawContextAccessMode = "normal"
    sparse_query_hints: list[SparseQueryHint] = Field(default_factory=list)
    query_type: QueryType = "default"
    retrieval_levels: list[int] = Field(default_factory=lambda: [0])
    # Wave 1 batch 2 (1-D): exact recall routing. The LLM decides; the
    # pipeline resolves deterministically.
    exact_recall_needed: bool = False
    exact_facets: list[ExactFacet] = Field(default_factory=list)

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

    @model_validator(mode="after")
    def validate_sparse_query_hints(self) -> "QueryIntelligenceResult":
        if not self.sparse_query_hints:
            return self
        sub_queries = set(self.sub_queries)
        seen_hint_targets: set[str] = set()
        normalized_hints: list[SparseQueryHint] = []
        for hint in self.sparse_query_hints:
            if hint.sub_query_text not in sub_queries:
                raise ValueError(
                    "SparseQueryHint.sub_query_text must reference an item from sub_queries"
                )
            if hint.sub_query_text in seen_hint_targets:
                raise ValueError(
                    "SparseQueryHint.sub_query_text values must be unique"
                )
            seen_hint_targets.add(hint.sub_query_text)
            normalized_hints.append(
                _normalize_sparse_hint_precision_fields(
                    hint,
                    query_type=self.query_type,
                    callback_bias=self.callback_bias,
                )
            )
        self.sparse_query_hints = normalized_hints
        if self.callback_bias:
            for hint in self.sparse_query_hints:
                if hint.quoted_phrases or hint.must_keep_terms:
                    continue
                raise ValueError(
                    "callback sparse_query_hints must preserve an explicit anchor "
                    "via quoted_phrases or must_keep_terms"
                )
        if self.query_type == "slot_fill":
            for hint in self.sparse_query_hints:
                if hint.quoted_phrases or hint.must_keep_terms:
                    continue
                raise ValueError(
                    "slot_fill sparse_query_hints must preserve concrete anchors "
                    "via quoted_phrases or must_keep_terms"
                )
        if self.query_type == "broad_list":
            for hint in self.sparse_query_hints:
                if hint.quoted_phrases or hint.must_keep_terms:
                    continue
                raise ValueError(
                    "broad_list sparse_query_hints must preserve explicit facet anchors "
                    "via quoted_phrases or must_keep_terms"
                )
        if self.query_type == "broad_list" and len(self.sparse_query_hints) > 1:
            signatures = [_sparse_hint_signature(hint) for hint in self.sparse_query_hints]
            if len(signatures) != len(set(signatures)):
                raise ValueError(
                    "broad_list sparse_query_hints must preserve distinct facet anchors "
                    "across sub_queries"
                )
        return self


class PlannedSubQuery(BaseModel):
    """Mechanical retrieval rewrites for one semantic sub-query."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    sparse_phrase: str | None = None
    quoted_phrases: list[str] = Field(default_factory=list)
    must_keep_terms: list[str] = Field(default_factory=list)
    fts_queries: list[str] = Field(min_length=1)

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


class RetrievalPlan(BaseModel):
    """Deterministic retrieval plan built before candidate search."""

    model_config = ConfigDict(extra="forbid")

    original_query: str | None = None
    assistant_mode_id: str
    workspace_id: str | None = None
    conversation_id: str
    fts_queries: list[str] = Field(default_factory=list)
    sub_query_plans: list[PlannedSubQuery] = Field(min_length=1)
    callback_bias: bool = False
    raw_context_access_mode: RawContextAccessMode = "normal"
    query_type: QueryType = "default"
    scope_filter: list[MemoryScope] = Field(default_factory=list)
    status_filter: list[MemoryStatus] = Field(default_factory=list)
    vector_limit: int = Field(ge=0, default=0)
    max_candidates: int = Field(ge=0)
    max_context_items: int = Field(gt=0)
    privacy_ceiling: int = Field(ge=0, le=3)
    retrieval_levels: list[int] = Field(default_factory=lambda: [0])
    temporal_query_range: TemporalQueryRange | None = None
    consequence_search_enabled: bool = False
    require_evidence_regrounding: bool = False
    need_driven_boosts: dict[NeedTrigger, float] = Field(default_factory=dict)
    skip_retrieval: bool = False
    # Wave 1 batch 2 (1-D): exact recall lane. When enabled, downstream
    # stages prioritize raw evidence and concrete L0 memories over
    # abstracted summaries/beliefs.
    exact_recall_mode: bool = False
    exact_facets: list[ExactFacet] = Field(default_factory=list)

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

    @field_validator("sub_query_plans")
    @classmethod
    def validate_sub_query_plans(cls, values: list[PlannedSubQuery]) -> list[PlannedSubQuery]:
        if not values:
            raise ValueError("sub_query_plans must not be empty")
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
    memory_category: MemoryCategory = MemoryCategory.UNKNOWN
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
    scope: MemoryScope
    dimension_name: str
    value_json: dict[str, Any] = Field(default_factory=dict)
    confidence: float
    source_memory_id: str
    updated_at: datetime


class MemoryConsentProfile(BaseModel):
    """Per-user consent counts for sensitive memory categories."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
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


class NeedDetectionTrace(BaseModel):
    """Trace of the need detection stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    detected_needs: list[str] = Field(default_factory=list)
    sub_queries: list[str] = Field(default_factory=list)
    sparse_hints: list[str] = Field(default_factory=list)
    query_type: str = "default"
    raw_context_access_mode: str = "normal"
    temporal_range: str | None = None
    retrieval_levels: list[int] = Field(default_factory=lambda: [0])
    degraded_mode: bool = False
    exact_recall_needed: bool = False
    exact_facets: list[str] = Field(default_factory=list)
    duration_ms: float = Field(ge=0.0)


class SubQuerySearchCount(BaseModel):
    """Per-sub-query candidate counts from search."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    subquery: str
    verbatim_pin: int = Field(ge=0, default=0)
    artifact_chunk: int = Field(ge=0, default=0)
    fts: int = Field(ge=0, default=0)
    embedding: int = Field(ge=0, default=0)
    raw_message: int = Field(ge=0, default=0)


class CandidateSearchTrace(BaseModel):
    """Trace of the candidate search stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fts_candidates_count: int = Field(ge=0, default=0)
    verbatim_pin_candidates_count: int = Field(ge=0, default=0)
    artifact_chunk_candidates_count: int = Field(ge=0, default=0)
    embedding_candidates_count: int = Field(ge=0, default=0)
    consequence_candidates_count: int = Field(ge=0, default=0)
    raw_message_candidates_count: int = Field(ge=0, default=0)
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
    top_score: float = 0.0
    median_score: float = 0.0
    min_score: float = 0.0
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
    duration_ms: float = Field(ge=0.0)


class RetrievalTrace(BaseModel):
    """Complete trace of a retrieval operation."""

    model_config = ConfigDict(extra="forbid")

    query_text: str
    user_id: str
    conversation_id: str
    timestamp_iso: str
    small_corpus_mode: bool = False
    degraded_mode: bool = False
    raw_context_access_mode: str = "normal"
    need_detection: NeedDetectionTrace | None = None
    candidate_search: CandidateSearchTrace | None = None
    scoring: ScoringTrace | None = None
    composition: CompositionTrace | None = None
    total_duration_ms: float = Field(ge=0.0, default=0.0)
