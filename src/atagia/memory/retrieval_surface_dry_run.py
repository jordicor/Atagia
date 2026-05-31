"""Dry-run generation of non-evidential persisted retrieval surfaces."""

from __future__ import annotations

from collections import Counter
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from atagia.core import json_utils
from atagia.core.clock import Clock
from atagia.core.config import Settings
from atagia.core.repositories import MemoryRetrievalSurfaceRepository
from atagia.models.schemas_memory import AliasKind, AnchorType
from atagia.services.llm_client import LLMClient, LLMCompletionRequest, LLMMessage
from atagia.services.model_resolution import resolve_component_model

RETRIEVAL_SURFACE_DRY_RUN_PROMPT_VERSION = "retrieval_surface_dry_run_v1"
RETRIEVAL_SURFACE_DRY_RUN_MAX_OUTPUT_TOKENS = 8192

SurfaceType = Literal["pivot", "anchor", "alias", "corpus_surface"]
VisibilityPolicy = Literal["base_memory_gated"]

_HIGH_RISK_ANCHOR_TYPES: set[str] = {
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


def _strip_required(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _strip_optional(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _validate_surface_semantic_contract(
    *,
    surface_type: SurfaceType,
    anchor_type: AnchorType | None,
    alias_kind: AliasKind | None,
    preserve_verbatim: bool,
    error_prefix: str = "",
) -> None:
    prefix = f"{error_prefix} " if error_prefix else ""
    if surface_type == "alias" and alias_kind is None:
        raise ValueError(f"{prefix}alias surfaces must include alias_kind")
    if surface_type == "anchor" and anchor_type is None:
        raise ValueError(f"{prefix}anchor surfaces must include anchor_type")
    if surface_type == "alias" and preserve_verbatim:
        raise ValueError(f"{prefix}alias surfaces cannot be preserve_verbatim")
    if anchor_type in _HIGH_RISK_ANCHOR_TYPES:
        if not preserve_verbatim:
            raise ValueError(f"{prefix}high-risk anchors must preserve verbatim")
        if surface_type == "alias" or alias_kind is not None:
            raise ValueError(f"{prefix}high-risk anchors cannot be broad aliases")


class RetrievalSurfaceSourceMemory(BaseModel):
    """Minimal source-memory snapshot for dry-run surface generation."""

    model_config = ConfigDict(extra="forbid")

    id: str
    user_id: str
    canonical_text: str
    index_text: str | None = None
    object_type: str | None = None
    language_codes: list[str] = Field(default_factory=list)
    privacy_level: int = Field(default=0, ge=0)
    sensitivity_level: int = Field(default=0, ge=0)

    @field_validator("id", "user_id", "canonical_text")
    @classmethod
    def validate_required_text(cls, value: str, info: Any) -> str:
        return _strip_required(value, str(info.field_name))

    @field_validator("index_text", "object_type")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        return _strip_optional(value)

    @field_validator("language_codes")
    @classmethod
    def validate_language_codes(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        for item in value:
            language_code = _strip_optional(item)
            if language_code is not None:
                normalized.append(language_code.lower())
        return normalized


class RetrievalSurfaceDraft(BaseModel):
    """Structured LLM-proposed retrieval surface before repository shaping."""

    model_config = ConfigDict(extra="forbid")

    memory_id: str
    surface_type: SurfaceType
    surface_text: str
    anchor_type: AnchorType | None = None
    alias_kind: AliasKind | None = None
    language_code: str | None = None
    preserve_verbatim: bool = False
    non_evidential: bool = True
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    visibility_policy: VisibilityPolicy = "base_memory_gated"
    evidence_claim: str | None = None
    derivation: dict[str, Any] = Field(default_factory=dict)

    @field_validator("memory_id", "surface_text")
    @classmethod
    def validate_required_text(cls, value: str, info: Any) -> str:
        return _strip_required(value, str(info.field_name))

    @field_validator("language_code", "evidence_claim")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        return _strip_optional(value)

    @field_validator("language_code")
    @classmethod
    def normalize_language_code(cls, value: str | None) -> str | None:
        return value.lower() if value is not None else None

    @field_validator("non_evidential")
    @classmethod
    def validate_non_evidential(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("persisted retrieval surfaces must be non_evidential")
        return value

    @model_validator(mode="after")
    def validate_surface_semantics(self) -> RetrievalSurfaceDraft:
        if self.evidence_claim is not None:
            raise ValueError("retrieval surfaces may not include evidence claims")
        _validate_surface_semantic_contract(
            surface_type=self.surface_type,
            anchor_type=self.anchor_type,
            alias_kind=self.alias_kind,
            preserve_verbatim=self.preserve_verbatim,
        )
        return self


class RetrievalSurfaceDryRunLLMResult(BaseModel):
    """Structured output contract returned by the surface generator LLM."""

    model_config = ConfigDict(extra="forbid")

    surfaces: list[RetrievalSurfaceDraft] = Field(default_factory=list)


class RetrievalSurfaceWouldWrite(BaseModel):
    """Repository-shaped persisted surface payload for review only."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    memory_id: str
    surface_type: SurfaceType
    surface_text: str
    anchor_type: AnchorType | None = None
    alias_kind: AliasKind | None = None
    language_code: str | None = None
    preserve_verbatim: bool
    non_evidential: Literal[True] = True
    confidence: float = Field(ge=0.0, le=1.0)
    visibility_policy: VisibilityPolicy = "base_memory_gated"
    base_privacy_level: int = Field(ge=0)
    base_sensitivity_level: int = Field(ge=0)
    derivation_kind: Literal["dry_run_llm"] = "dry_run_llm"
    derivation_model: str
    derivation_prompt_version: str
    derivation_json: dict[str, Any] = Field(default_factory=dict)
    status: Literal["active"] = "active"
    dry_run_only: Literal[True] = True


class RetrievalSurfaceDryRunReport(BaseModel):
    """Auditable dry-run report; nothing in this report has been persisted."""

    model_config = ConfigDict(extra="forbid")

    dry_run: Literal[True] = True
    writes_enabled: Literal[False] = False
    generated_at: str
    prompt_version: str
    derivation_model: str
    source_memory_count: int
    surface_count: int
    surfaces: list[RetrievalSurfaceWouldWrite] = Field(default_factory=list)


class RetrievalSurfaceStorageCounts(BaseModel):
    """Surface-table counts used to prove a dry-run did not write."""

    model_config = ConfigDict(extra="forbid")

    memory_retrieval_surfaces: int = Field(ge=0)
    memory_retrieval_surfaces_fts: int = Field(ge=0)
    active_joined_surfaces: int = Field(ge=0)


class RetrievalSurfaceBackfillDryRunMemory(BaseModel):
    """Content-minimal selected memory metadata for a local backfill dry-run."""

    model_config = ConfigDict(extra="forbid")

    selection_kind: str
    id: str
    user_id: str
    object_type: str
    scope: str
    status: str
    privacy_level: int = Field(ge=0)
    sensitivity: str
    language_codes: list[str] = Field(default_factory=list)


class RetrievalSurfaceBackfillDryRunReport(BaseModel):
    """Auditable small-batch dry-run report for copied local DBs."""

    model_config = ConfigDict(extra="forbid")

    dry_run: Literal[True] = True
    writes_enabled: Literal[False] = False
    database_label: str | None = None
    selected_memory_count: int = Field(ge=0)
    selected_memories: list[RetrievalSurfaceBackfillDryRunMemory]
    before_counts: RetrievalSurfaceStorageCounts
    after_counts: RetrievalSurfaceStorageCounts
    no_write_verified: bool
    dry_run_report: RetrievalSurfaceDryRunReport
    proposed_surface_count: int = Field(ge=0)
    surface_type_counts: dict[str, int] = Field(default_factory=dict)
    language_counts: dict[str, int] = Field(default_factory=dict)
    rejected_surface_count: int = Field(default=0, ge=0)
    high_risk_source_memory_count: int = Field(ge=0)
    high_risk_surface_count: int = Field(ge=0)


class RetrievalSurfaceApprovedWrite(BaseModel):
    """Approved persisted-surface payload that may be written explicitly."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    memory_id: str
    surface_type: SurfaceType
    surface_text: str
    anchor_type: AnchorType | None = None
    alias_kind: AliasKind | None = None
    language_code: str | None = None
    preserve_verbatim: bool
    non_evidential: Literal[True] = True
    confidence: float = Field(ge=0.0, le=1.0)
    visibility_policy: VisibilityPolicy = "base_memory_gated"
    base_privacy_level: int = Field(ge=0)
    base_sensitivity_level: int = Field(ge=0)
    derivation_kind: Literal["dry_run_llm"] = "dry_run_llm"
    derivation_model: str
    derivation_prompt_version: str
    derivation_json: dict[str, Any] = Field(default_factory=dict)
    status: Literal["active"] = "active"
    dry_run_only: Literal[False] = False
    approved_for_write: Literal[True] = True
    approval_id: str
    approved_at: str
    approved_by: str | None = None
    approval_note: str | None = None

    @field_validator("user_id", "memory_id", "surface_text", "approval_id", "approved_at")
    @classmethod
    def validate_required_text(cls, value: str, info: Any) -> str:
        return _strip_required(value, str(info.field_name))

    @field_validator("language_code", "approved_by", "approval_note")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        return _strip_optional(value)

    @field_validator("language_code")
    @classmethod
    def normalize_language_code(cls, value: str | None) -> str | None:
        return value.lower() if value is not None else None

    @model_validator(mode="after")
    def validate_approved_surface(self) -> RetrievalSurfaceApprovedWrite:
        _validate_surface_semantic_contract(
            surface_type=self.surface_type,
            anchor_type=self.anchor_type,
            alias_kind=self.alias_kind,
            preserve_verbatim=self.preserve_verbatim,
            error_prefix="approved",
        )
        return self

    @classmethod
    def from_reviewed_surface(
        cls,
        surface: RetrievalSurfaceWouldWrite,
        *,
        approval_id: str,
        approved_at: str,
        approved_by: str | None = None,
        approval_note: str | None = None,
    ) -> RetrievalSurfaceApprovedWrite:
        return cls(
            **surface.model_dump(exclude={"dry_run_only"}),
            dry_run_only=False,
            approved_for_write=True,
            approval_id=approval_id,
            approved_at=approved_at,
            approved_by=approved_by,
            approval_note=approval_note,
        )


class RetrievalSurfaceWriteResult(BaseModel):
    """Single persisted-surface write result."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    memory_id: str
    surface_id: str
    surface_type: SurfaceType
    surface_text: str
    status: str


class RetrievalSurfaceWriteReport(BaseModel):
    """Auditable opt-in write report for reviewed retrieval surfaces."""

    model_config = ConfigDict(extra="forbid")

    dry_run: bool
    writes_enabled: bool
    requested_surface_count: int
    written_surface_count: int
    skipped_surface_count: int
    results: list[RetrievalSurfaceWriteResult] = Field(default_factory=list)


class RetrievalSurfaceWriter:
    """Opt-in writer for already-reviewed retrieval surface payloads."""

    def __init__(
        self,
        repository: MemoryRetrievalSurfaceRepository,
        clock: Clock,
    ) -> None:
        self._repository = repository
        self._clock = clock

    async def write_approved(
        self,
        surfaces: Sequence[RetrievalSurfaceApprovedWrite | dict[str, Any]],
        *,
        enable_write: bool = False,
    ) -> RetrievalSurfaceWriteReport:
        approved_surfaces = [
            RetrievalSurfaceApprovedWrite.model_validate(surface)
            for surface in surfaces
        ]
        if not enable_write:
            return RetrievalSurfaceWriteReport(
                dry_run=True,
                writes_enabled=False,
                requested_surface_count=len(approved_surfaces),
                written_surface_count=0,
                skipped_surface_count=len(approved_surfaces),
                results=[],
            )

        results: list[RetrievalSurfaceWriteResult] = []
        for surface in approved_surfaces:
            row = await self._repository.upsert_surface(
                user_id=surface.user_id,
                memory_id=surface.memory_id,
                surface_type=surface.surface_type,
                surface_text=surface.surface_text,
                anchor_type=surface.anchor_type,
                alias_kind=surface.alias_kind,
                language_code=surface.language_code,
                preserve_verbatim=surface.preserve_verbatim,
                non_evidential=True,
                confidence=surface.confidence,
                derivation_kind=surface.derivation_kind,
                derivation_model=surface.derivation_model,
                derivation_prompt_version=surface.derivation_prompt_version,
                derivation=self._derivation_for_write(surface),
                status=surface.status,
            )
            results.append(
                RetrievalSurfaceWriteResult(
                    user_id=str(row["user_id"]),
                    memory_id=str(row["memory_id"]),
                    surface_id=str(row["id"]),
                    surface_type=row["surface_type"],
                    surface_text=str(row["surface_text"]),
                    status=str(row["status"]),
                )
            )
        return RetrievalSurfaceWriteReport(
            dry_run=False,
            writes_enabled=True,
            requested_surface_count=len(approved_surfaces),
            written_surface_count=len(results),
            skipped_surface_count=0,
            results=results,
        )

    def _derivation_for_write(
        self,
        surface: RetrievalSurfaceApprovedWrite,
    ) -> dict[str, Any]:
        return {
            **surface.derivation_json,
            "dry_run": False,
            "written_from_dry_run": True,
            "written_at": self._clock.now().isoformat(),
            "approval": {
                "approval_id": surface.approval_id,
                "approved_at": surface.approved_at,
                "approved_by": surface.approved_by,
                "approval_note": surface.approval_note,
            },
        }


class RetrievalSurfaceDryRunGenerator:
    """Plan persisted retrieval surfaces without writing them."""

    def __init__(
        self,
        llm_client: LLMClient[Any],
        clock: Clock,
        settings: Settings | None = None,
        *,
        model: str | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._clock = clock
        resolved_settings = settings or Settings.from_env()
        self._model = model or resolve_component_model(
            resolved_settings,
            "coverage_expander",
        )

    async def generate(
        self,
        memories: Sequence[RetrievalSurfaceSourceMemory | dict[str, Any]],
    ) -> RetrievalSurfaceDryRunReport:
        source_memories = [
            RetrievalSurfaceSourceMemory.model_validate(memory)
            for memory in memories
        ]
        generated_at = self._clock.now().isoformat()
        if not source_memories:
            return RetrievalSurfaceDryRunReport(
                generated_at=generated_at,
                prompt_version=RETRIEVAL_SURFACE_DRY_RUN_PROMPT_VERSION,
                derivation_model=self._model,
                source_memory_count=0,
                surface_count=0,
                surfaces=[],
            )

        request = LLMCompletionRequest(
            model=self._model,
            messages=[
                LLMMessage(
                    role="system",
                    content=(
                        "Produce dry-run persisted retrieval surfaces as "
                        "structured JSON only."
                    ),
                ),
                LLMMessage(role="user", content=self._build_prompt(source_memories)),
            ],
            max_output_tokens=RETRIEVAL_SURFACE_DRY_RUN_MAX_OUTPUT_TOKENS,
            response_schema=RetrievalSurfaceDryRunLLMResult.model_json_schema(),
            metadata={
                "purpose": "retrieval_surface_generation_dry_run",
                "prompt_version": RETRIEVAL_SURFACE_DRY_RUN_PROMPT_VERSION,
                "source_memory_count": len(source_memories),
                "dry_run": True,
            },
        )
        llm_result = await self._llm_client.complete_structured(
            request,
            RetrievalSurfaceDryRunLLMResult,
        )
        surfaces = self._shape_would_write_payloads(
            source_memories=source_memories,
            llm_result=llm_result,
            generated_at=generated_at,
        )
        return RetrievalSurfaceDryRunReport(
            generated_at=generated_at,
            prompt_version=RETRIEVAL_SURFACE_DRY_RUN_PROMPT_VERSION,
            derivation_model=self._model,
            source_memory_count=len(source_memories),
            surface_count=len(surfaces),
            surfaces=surfaces,
        )

    def _shape_would_write_payloads(
        self,
        *,
        source_memories: list[RetrievalSurfaceSourceMemory],
        llm_result: RetrievalSurfaceDryRunLLMResult,
        generated_at: str,
    ) -> list[RetrievalSurfaceWouldWrite]:
        memories_by_id = {memory.id: memory for memory in source_memories}
        surfaces: list[RetrievalSurfaceWouldWrite] = []
        for index, draft in enumerate(llm_result.surfaces):
            source_memory = memories_by_id.get(draft.memory_id)
            if source_memory is None:
                raise ValueError(
                    f"retrieval surface references unknown memory_id: {draft.memory_id}"
                )
            surfaces.append(
                RetrievalSurfaceWouldWrite(
                    user_id=source_memory.user_id,
                    memory_id=source_memory.id,
                    surface_type=draft.surface_type,
                    surface_text=draft.surface_text,
                    anchor_type=draft.anchor_type,
                    alias_kind=draft.alias_kind,
                    language_code=draft.language_code,
                    preserve_verbatim=draft.preserve_verbatim,
                    non_evidential=True,
                    confidence=draft.confidence,
                    visibility_policy=draft.visibility_policy,
                    base_privacy_level=source_memory.privacy_level,
                    base_sensitivity_level=source_memory.sensitivity_level,
                    derivation_model=self._model,
                    derivation_prompt_version=RETRIEVAL_SURFACE_DRY_RUN_PROMPT_VERSION,
                    derivation_json={
                        "dry_run": True,
                        "generated_at": generated_at,
                        "llm_output_index": index,
                        "source_memory_id": source_memory.id,
                        "source_memory_language_codes": source_memory.language_codes,
                        "source_memory_privacy_level": source_memory.privacy_level,
                        "source_memory_sensitivity_level": source_memory.sensitivity_level,
                        "visibility_policy": draft.visibility_policy,
                        "llm_derivation": draft.derivation,
                    },
                )
            )
        return surfaces

    def _build_prompt(
        self,
        source_memories: list[RetrievalSurfaceSourceMemory],
    ) -> str:
        memory_payload = [
            {
                "id": memory.id,
                "user_id": memory.user_id,
                "canonical_text": memory.canonical_text,
                "index_text": memory.index_text,
                "object_type": memory.object_type,
                "language_codes": memory.language_codes,
                "privacy_level": memory.privacy_level,
                "sensitivity_level": memory.sensitivity_level,
            }
            for memory in source_memories
        ]
        return "\n".join(
            [
                "Plan persisted retrieval surfaces for a dry-run review.",
                "Return JSON only, matching the provided schema exactly.",
                "Do not include markdown fences, preambles, tags, or explanations.",
                "This is dry-run only. Do not imply that anything has been written.",
                "Surfaces are non-evidential retrieval aids. They may help retrieve a base memory, but they never prove an answer.",
                "Set non_evidential=true for every surface.",
                "Set visibility_policy exactly to base_memory_gated for every surface.",
                "Do not request broader visibility, lower privacy, or policy overrides.",
                "Use only generic surface types: pivot, anchor, alias, corpus_surface.",
                "Use only generic anchor and alias kinds from the schema.",
                "For proper names, people, organizations, locations, codes, quantities, dates, addresses, and quoted phrases: preserve the original surface verbatim, set preserve_verbatim=true, and do not create broad aliases.",
                "For common concepts and attributes, aliases may include translations, transliterations, spelling variants, acronym expansions, domain synonyms, or visible corpus surfaces.",
                "Do not create medication-specific, benchmark-specific, or domain-specific fields.",
                "Do not use or imply regexes, keyword lists, or hardcoded semantic classifiers; classification must be represented in the structured fields.",
                "Do not include evidence claims. If a surface is not directly grounded in the supplied memory, omit it.",
                f"prompt_version={RETRIEVAL_SURFACE_DRY_RUN_PROMPT_VERSION}",
                "<source_memories>",
                json_utils.dumps(memory_payload, indent=2, sort_keys=True),
                "</source_memories>",
            ]
        )


async def run_small_local_backfill_surface_dry_run(
    connection: Any,
    generator: RetrievalSurfaceDryRunGenerator,
    *,
    user_id: str,
    limit: int = 3,
    database_label: str | None = None,
) -> RetrievalSurfaceBackfillDryRunReport:
    """Run a no-write surface dry-run over a small copied local DB sample."""

    if limit < 0:
        raise ValueError("limit must be non-negative")
    user_id = _strip_required(user_id, "user_id")
    before_counts = await _surface_storage_counts(connection)
    source_memories, selected_memories = await _select_backfill_dry_run_memories(
        connection,
        user_id=user_id,
        limit=limit,
    )
    dry_run_report = await generator.generate(source_memories)
    after_counts = await _surface_storage_counts(connection)
    surface_type_counts = Counter(
        surface.surface_type for surface in dry_run_report.surfaces
    )
    language_counts = Counter(
        surface.language_code or "unknown" for surface in dry_run_report.surfaces
    )
    high_risk_source_memory_count = sum(
        1
        for memory in selected_memories
        if _is_high_risk_metadata(memory.privacy_level, memory.sensitivity)
    )
    high_risk_surface_count = sum(
        1
        for surface in dry_run_report.surfaces
        if surface.preserve_verbatim
        or surface.base_privacy_level >= 2
        or surface.base_sensitivity_level >= 2
    )
    return RetrievalSurfaceBackfillDryRunReport(
        database_label=database_label,
        selected_memory_count=len(selected_memories),
        selected_memories=selected_memories,
        before_counts=before_counts,
        after_counts=after_counts,
        no_write_verified=before_counts == after_counts
        and dry_run_report.writes_enabled is False,
        dry_run_report=dry_run_report,
        proposed_surface_count=dry_run_report.surface_count,
        surface_type_counts=dict(surface_type_counts),
        language_counts=dict(language_counts),
        rejected_surface_count=0,
        high_risk_source_memory_count=high_risk_source_memory_count,
        high_risk_surface_count=high_risk_surface_count,
    )


async def _surface_storage_counts(connection: Any) -> RetrievalSurfaceStorageCounts:
    surface_count = await _fetch_count(
        connection,
        "SELECT COUNT(*) AS count FROM memory_retrieval_surfaces",
    )
    fts_count = await _fetch_count(
        connection,
        "SELECT COUNT(*) AS count FROM memory_retrieval_surfaces_fts",
    )
    active_joined_count = await _fetch_count(
        connection,
        """
        SELECT COUNT(*) AS count
        FROM memory_retrieval_surfaces AS mrs
        JOIN memory_objects AS mo
          ON mo.id = mrs.memory_id
         AND mo.user_id = mrs.user_id
        WHERE mrs.status = 'active'
          AND mo.status = 'active'
        """,
    )
    return RetrievalSurfaceStorageCounts(
        memory_retrieval_surfaces=surface_count,
        memory_retrieval_surfaces_fts=fts_count,
        active_joined_surfaces=active_joined_count,
    )


async def _fetch_count(
    connection: Any,
    sql: str,
    parameters: Sequence[Any] = (),
) -> int:
    cursor = await connection.execute(sql, tuple(parameters))
    row = await cursor.fetchone()
    return int(row["count"] if "count" in row.keys() else row[0])


async def _select_backfill_dry_run_memories(
    connection: Any,
    *,
    user_id: str,
    limit: int,
) -> tuple[list[RetrievalSurfaceSourceMemory], list[RetrievalSurfaceBackfillDryRunMemory]]:
    if limit == 0:
        return [], []

    selected_rows: list[tuple[str, Any]] = []
    seen_ids: set[str] = set()
    for selection_kind, predicate in [
        (
            "ordinary_memory",
            """
            object_type != 'summary_view'
            AND privacy_level <= 1
            AND LOWER(COALESCE(sensitivity, 'unknown')) NOT IN ('private', 'secret')
            """,
        ),
        ("summary_mirror", "object_type = 'summary_view'"),
        (
            "high_risk_privacy_sensitive",
            """
            privacy_level >= 2
            OR LOWER(COALESCE(sensitivity, 'unknown')) IN ('private', 'secret')
            """,
        ),
    ]:
        if len(selected_rows) >= limit:
            break
        row = await _fetch_backfill_memory_row(
            connection,
            user_id=user_id,
            predicate=predicate,
            exclude_ids=seen_ids,
        )
        if row is None:
            continue
        memory_id = str(row["id"])
        seen_ids.add(memory_id)
        selected_rows.append((selection_kind, row))

    while len(selected_rows) < limit:
        row = await _fetch_backfill_memory_row(
            connection,
            user_id=user_id,
            predicate="1 = 1",
            exclude_ids=seen_ids,
        )
        if row is None:
            break
        memory_id = str(row["id"])
        seen_ids.add(memory_id)
        selected_rows.append(("fallback", row))

    source_memories: list[RetrievalSurfaceSourceMemory] = []
    selected_memories: list[RetrievalSurfaceBackfillDryRunMemory] = []
    for selection_kind, row in selected_rows:
        language_codes = _decode_language_codes(row["language_codes_json"])
        sensitivity = str(row["sensitivity"] or "unknown")
        source_memories.append(
            RetrievalSurfaceSourceMemory(
                id=str(row["id"]),
                user_id=str(row["user_id"]),
                canonical_text=str(row["canonical_text"]),
                index_text=row["index_text"],
                object_type=row["object_type"],
                language_codes=language_codes,
                privacy_level=int(row["privacy_level"]),
                sensitivity_level=_sensitivity_level(sensitivity),
            )
        )
        selected_memories.append(
            RetrievalSurfaceBackfillDryRunMemory(
                selection_kind=selection_kind,
                id=str(row["id"]),
                user_id=str(row["user_id"]),
                object_type=str(row["object_type"]),
                scope=str(row["scope"]),
                status=str(row["status"]),
                privacy_level=int(row["privacy_level"]),
                sensitivity=sensitivity,
                language_codes=language_codes,
            )
        )
    return source_memories, selected_memories


async def _fetch_backfill_memory_row(
    connection: Any,
    *,
    user_id: str,
    predicate: str,
    exclude_ids: set[str],
) -> Any | None:
    clauses = ["status = 'active'", "user_id = ?", f"({predicate})"]
    parameters: list[Any] = [user_id]
    if exclude_ids:
        placeholders = ", ".join("?" for _ in exclude_ids)
        clauses.append(f"id NOT IN ({placeholders})")
        parameters.extend(sorted(exclude_ids))
    cursor = await connection.execute(
        f"""
        SELECT id,
               user_id,
               object_type,
               scope,
               canonical_text,
               index_text,
               language_codes_json,
               privacy_level,
               sensitivity,
               status,
               updated_at
        FROM memory_objects
        WHERE {" AND ".join(clauses)}
        ORDER BY updated_at DESC, id ASC
        LIMIT 1
        """,
        tuple(parameters),
    )
    return await cursor.fetchone()


def _decode_language_codes(value: Any) -> list[str]:
    if not value:
        return []
    try:
        parsed = json_utils.loads(value)
    except json_utils.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    language_codes: list[str] = []
    for item in parsed:
        language_code = _strip_optional(str(item))
        if language_code is not None:
            language_codes.append(language_code.lower())
    return language_codes


def _sensitivity_level(value: str) -> int:
    return {
        "public": 0,
        "unknown": 1,
        "private": 2,
        "secret": 3,
    }.get(value.lower(), 1)


def _is_high_risk_metadata(privacy_level: int, sensitivity: str) -> bool:
    return privacy_level >= 2 or _sensitivity_level(sensitivity) >= 2
