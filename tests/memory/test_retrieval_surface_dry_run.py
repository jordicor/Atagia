"""Tests for dry-run persisted retrieval surface generation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from atagia.core import json_utils
from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import (
    MemoryObjectRepository,
    MemoryRetrievalSurfaceRepository,
    UserRepository,
)
from atagia.memory.candidate_search import CandidateSearch
from atagia.memory.context_composer import ContextComposer
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.memory.policy_manifest import PolicyResolver
from atagia.memory.retrieval_surface_dry_run import (
    RETRIEVAL_SURFACE_DRY_RUN_PROMPT_VERSION,
    RetrievalSurfaceApprovedWrite,
    RetrievalSurfaceDryRunGenerator,
    RetrievalSurfaceWriter,
    run_small_local_backfill_surface_dry_run,
)
from atagia.models.schemas_memory import (
    MemorySensitivity,
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    MemoryStatus,
    PlannedSubQuery,
    RetrievalPlan,
    ScoredCandidate,
)
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMProvider,
    StructuredOutputError,
)

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"
CLOCK = FrozenClock(datetime(2026, 5, 14, 12, 0, tzinfo=timezone.utc))


def _resolved_policy():
    manifest = ManifestLoader(MANIFESTS_DIR).load_all()["coding_debug"]
    return PolicyResolver().resolve(manifest, None, None)


def _persisted_surface_plan(
    fts_query: str,
    *,
    query_type: str = "slot_fill",
    exact_recall_mode: bool = True,
) -> RetrievalPlan:
    return RetrievalPlan(
        original_query=fts_query,
        assistant_mode_id="coding_debug",
        conversation_id="cnv_1",
        sub_query_plans=[
            PlannedSubQuery(
                text=fts_query,
                fts_queries=[fts_query],
                fts_query_kinds=["surface_probe"],
            )
        ],
        scope_filter=[MemoryScope.GLOBAL_USER],
        status_filter=[MemoryStatus.ACTIVE],
        query_type=query_type,
        max_candidates=10,
        max_context_items=5,
        privacy_ceiling=1,
        retrieval_levels=[0],
        exact_recall_mode=exact_recall_mode,
    )


class SurfaceDryRunProvider(LLMProvider):
    name = "openai"

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.requests: list[LLMCompletionRequest] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.requests.append(request)
        return LLMCompletionResponse(
            provider=self.name,
            model=request.model,
            output_text=json_utils.dumps(self.payload),
        )


def _generator(
    payload: dict[str, Any],
) -> tuple[RetrievalSurfaceDryRunGenerator, SurfaceDryRunProvider]:
    provider = SurfaceDryRunProvider(payload)
    client = LLMClient(
        providers=[provider],
        structured_output_retry_attempts=0,
    )
    return (
        RetrievalSurfaceDryRunGenerator(
            client,
            CLOCK,
            model="openai/test-model",
        ),
        provider,
    )


def _memory(
    memory_id: str = "mem_1",
    *,
    canonical_text: str = "Ben's new apartment address is 120 Maple Street.",
    language_codes: list[str] | None = None,
    privacy_level: int = 0,
    sensitivity_level: int = 0,
) -> dict[str, Any]:
    return {
        "id": memory_id,
        "user_id": "usr_1",
        "canonical_text": canonical_text,
        "index_text": None,
        "object_type": "evidence",
        "language_codes": language_codes or ["en"],
        "privacy_level": privacy_level,
        "sensitivity_level": sensitivity_level,
    }


def _surface(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "memory_id": "mem_1",
        "surface_type": "alias",
        "surface_text": "direccion del apartamento",
        "alias_kind": "translation",
        "language_code": "es",
        "preserve_verbatim": False,
        "non_evidential": True,
        "confidence": 0.82,
        "visibility_policy": "base_memory_gated",
        "derivation": {"reason": "Spanish retrieval bridge"},
    }
    payload.update(overrides)
    return payload


@pytest.mark.asyncio
async def test_dry_run_validates_would_write_payloads_without_writing() -> None:
    generator, provider = _generator(
        {
            "surfaces": [
                _surface(),
                _surface(
                    memory_id="mem_2",
                    surface_type="pivot",
                    surface_text="new apartment address",
                    alias_kind=None,
                    language_code="en",
                    derivation={"reason": "English pivot for Spanish memory"},
                ),
            ]
        }
    )

    report = await generator.generate(
        [
            _memory(),
            _memory(
                "mem_2",
                canonical_text="La direccion nueva del apartamento es 120 Maple Street.",
                language_codes=["es"],
            ),
        ]
    )

    assert report.dry_run is True
    assert report.writes_enabled is False
    assert report.prompt_version == RETRIEVAL_SURFACE_DRY_RUN_PROMPT_VERSION
    assert report.derivation_model == "openai/test-model"
    assert report.source_memory_count == 2
    assert report.surface_count == 2
    assert len(report.surfaces) == 2
    surface = report.surfaces[0]
    assert surface.user_id == "usr_1"
    assert surface.memory_id == "mem_1"
    assert surface.surface_type == "alias"
    assert surface.surface_text == "direccion del apartamento"
    assert surface.non_evidential is True
    assert surface.visibility_policy == "base_memory_gated"
    assert surface.derivation_kind == "dry_run_llm"
    assert surface.derivation_model == "openai/test-model"
    assert surface.derivation_prompt_version == RETRIEVAL_SURFACE_DRY_RUN_PROMPT_VERSION
    assert surface.derivation_json["dry_run"] is True
    assert surface.derivation_json["source_memory_id"] == "mem_1"
    assert surface.dry_run_only is True
    assert provider.requests[0].metadata["purpose"] == "retrieval_surface_generation_dry_run"
    assert provider.requests[0].metadata["source_memory_count"] == 2
    assert provider.requests[0].response_schema is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_surface",
    [
        _surface(non_evidential=False),
        _surface(evidence_claim="This surface proves the address."),
    ],
)
async def test_dry_run_rejects_evidential_or_claiming_outputs(
    invalid_surface: dict[str, Any],
) -> None:
    generator, _provider = _generator({"surfaces": [invalid_surface]})

    with pytest.raises(StructuredOutputError):
        await generator.generate([_memory()])


@pytest.mark.asyncio
async def test_dry_run_preserves_high_risk_literals_without_broad_aliases() -> None:
    valid_high_risk_anchor = _surface(
        surface_type="anchor",
        surface_text="SA42",
        anchor_type="code",
        alias_kind=None,
        language_code=None,
        preserve_verbatim=True,
        derivation={"reason": "Exact code anchor"},
    )
    generator, _provider = _generator({"surfaces": [valid_high_risk_anchor]})

    report = await generator.generate([_memory()])

    assert report.surfaces[0].surface_type == "anchor"
    assert report.surfaces[0].anchor_type == "code"
    assert report.surfaces[0].preserve_verbatim is True
    assert report.surfaces[0].alias_kind is None

    broad_alias_for_code = _surface(
        surface_type="alias",
        surface_text="SA43",
        anchor_type="code",
        alias_kind="spelling_variant",
        preserve_verbatim=False,
    )
    generator, _provider = _generator({"surfaces": [broad_alias_for_code]})

    with pytest.raises(StructuredOutputError):
        await generator.generate([_memory()])


@pytest.mark.asyncio
async def test_dry_run_keeps_privacy_sensitive_surfaces_base_gated() -> None:
    generator, _provider = _generator({"surfaces": [_surface(surface_type="pivot", alias_kind=None)]})

    report = await generator.generate([_memory(privacy_level=3, sensitivity_level=2)])

    surface = report.surfaces[0]
    assert surface.visibility_policy == "base_memory_gated"
    assert surface.base_privacy_level == 3
    assert surface.base_sensitivity_level == 2
    assert surface.derivation_json["source_memory_privacy_level"] == 3
    assert surface.derivation_json["visibility_policy"] == "base_memory_gated"

    broader_visibility = _surface(visibility_policy="public")
    generator, _provider = _generator({"surfaces": [broader_visibility]})

    with pytest.raises(StructuredOutputError):
        await generator.generate([_memory(privacy_level=3)])


@pytest.mark.asyncio
async def test_small_local_backfill_dry_run_selects_sample_without_writing() -> None:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    try:
        await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), CLOCK)
        users = UserRepository(connection, CLOCK)
        memories = MemoryObjectRepository(connection, CLOCK)
        surfaces = MemoryRetrievalSurfaceRepository(connection, CLOCK)
        await users.create_user("usr_1")
        await memories.create_memory_object(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CHAT,
            canonical_text="The payment worker previously had connection pool instability.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.9,
            privacy_level=0,
            sensitivity=MemorySensitivity.PUBLIC,
            language_codes=["en"],
            memory_id="mem_ordinary",
        )
        await memories.create_memory_object(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.SUMMARY_VIEW,
            scope=MemoryScope.USER,
            canonical_text="The user has apartment lease logistics and reminder tasks.",
            source_kind=MemorySourceKind.SUMMARIZED,
            confidence=0.9,
            privacy_level=0,
            sensitivity=MemorySensitivity.PUBLIC,
            language_codes=["en"],
            memory_id="mem_summary",
        )
        await memories.create_memory_object(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.CHAT,
            canonical_text="The user said the 6 AM routine improved anxiety.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.9,
            privacy_level=3,
            sensitivity=MemorySensitivity.SECRET,
            language_codes=["en"],
            memory_id="mem_secret",
        )
        await surfaces.upsert_surface(
            user_id="usr_1",
            memory_id="mem_ordinary",
            surface_type="pivot",
            surface_text="existing diagnostic seed",
            preserve_verbatim=False,
            non_evidential=True,
            confidence=0.5,
            derivation_kind="manual_fixture",
        )
        generator, provider = _generator(
            {
                "surfaces": [
                    _surface(
                        memory_id="mem_ordinary",
                        surface_text="payment worker stability",
                        alias_kind="domain_synonym",
                        language_code="en",
                    ),
                    _surface(
                        memory_id="mem_summary",
                        surface_type="pivot",
                        surface_text="apartment lease logistics",
                        alias_kind=None,
                        language_code="en",
                    ),
                    _surface(
                        memory_id="mem_secret",
                        surface_type="anchor",
                        surface_text="6 AM",
                        anchor_type="date_time",
                        alias_kind=None,
                        language_code=None,
                        preserve_verbatim=True,
                    ),
                ]
            }
        )

        report = await run_small_local_backfill_surface_dry_run(
            connection,
            generator,
            user_id="usr_1",
            database_label="local-test-copy",
        )

        assert report.dry_run is True
        assert report.writes_enabled is False
        assert report.database_label == "local-test-copy"
        assert report.selected_memory_count == 3
        assert {memory.selection_kind for memory in report.selected_memories} == {
            "ordinary_memory",
            "summary_mirror",
            "high_risk_privacy_sensitive",
        }
        assert report.before_counts == report.after_counts
        assert report.before_counts.memory_retrieval_surfaces == 1
        assert report.before_counts.memory_retrieval_surfaces_fts == 1
        assert report.no_write_verified is True
        assert report.dry_run_report.source_memory_count == 3
        assert report.proposed_surface_count == 3
        assert report.surface_type_counts == {"alias": 1, "pivot": 1, "anchor": 1}
        assert report.language_counts == {"en": 2, "unknown": 1}
        assert report.rejected_surface_count == 0
        assert report.high_risk_source_memory_count == 1
        assert report.high_risk_surface_count == 1
        assert provider.requests[0].metadata["dry_run"] is True

        rows = await surfaces.list_surfaces_for_memory(
            user_id="usr_1",
            memory_id="mem_ordinary",
        )
        assert len(rows) == 1
        assert rows[0]["surface_text"] == "existing diagnostic seed"
    finally:
        await connection.close()


async def _surface_runtime():
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    await sync_assistant_modes(connection, ManifestLoader(MANIFESTS_DIR).load_all(), CLOCK)
    users = UserRepository(connection, CLOCK)
    memories = MemoryObjectRepository(connection, CLOCK)
    surfaces = MemoryRetrievalSurfaceRepository(connection, CLOCK)
    await users.create_user("usr_1")
    await memories.create_memory_object(
        user_id="usr_1",
        assistant_mode_id="coding_debug",
        object_type=MemoryObjectType.EVIDENCE,
        scope=MemoryScope.GLOBAL_USER,
        canonical_text="Ben's new apartment address is 120 Maple Street.",
        source_kind=MemorySourceKind.EXTRACTED,
        confidence=0.9,
        privacy_level=0,
        memory_id="mem_1",
    )
    return connection, surfaces


async def _approved_surface() -> RetrievalSurfaceApprovedWrite:
    generator, _provider = _generator(
        {
            "surfaces": [
                _surface(
                    surface_type="anchor",
                    surface_text="120 Maple Street",
                    anchor_type="address",
                    alias_kind=None,
                    language_code="en",
                    preserve_verbatim=True,
                    confidence=0.91,
                    derivation={"reason": "Exact address anchor"},
                )
            ]
        }
    )
    report = await generator.generate([_memory()])
    return RetrievalSurfaceApprovedWrite.from_reviewed_surface(
        report.surfaces[0],
        approval_id="approval_surface_1",
        approved_at=CLOCK.now().isoformat(),
        approved_by="reviewer",
        approval_note="Synthetic fixture approval.",
    )


@pytest.mark.asyncio
async def test_surface_writer_requires_explicit_enable_write() -> None:
    connection, surfaces = await _surface_runtime()
    try:
        approved = await _approved_surface()
        writer = RetrievalSurfaceWriter(surfaces, CLOCK)

        disabled_report = await writer.write_approved([approved])

        assert disabled_report.dry_run is True
        assert disabled_report.writes_enabled is False
        assert disabled_report.written_surface_count == 0
        assert disabled_report.skipped_surface_count == 1
        assert await surfaces.list_surfaces_for_memory(
            user_id="usr_1",
            memory_id="mem_1",
        ) == []

        write_report = await writer.write_approved([approved], enable_write=True)

        assert write_report.dry_run is False
        assert write_report.writes_enabled is True
        assert write_report.written_surface_count == 1
        assert write_report.results[0].memory_id == "mem_1"
        rows = await surfaces.list_surfaces_for_memory(
            user_id="usr_1",
            memory_id="mem_1",
        )
        assert len(rows) == 1
        row = rows[0]
        assert row["surface_type"] == "anchor"
        assert row["surface_text"] == "120 Maple Street"
        assert row["anchor_type"] == "address"
        assert row["alias_kind"] is None
        assert row["language_code"] == "en"
        assert row["preserve_verbatim"] == 1
        assert row["non_evidential"] == 1
        assert row["confidence"] == pytest.approx(0.91)
        assert row["derivation_kind"] == "dry_run_llm"
        assert row["derivation_model"] == "openai/test-model"
        assert row["derivation_prompt_version"] == RETRIEVAL_SURFACE_DRY_RUN_PROMPT_VERSION
        assert row["derivation_json"]["written_from_dry_run"] is True
        assert row["derivation_json"]["visibility_policy"] == "base_memory_gated"
        assert row["derivation_json"]["approval"]["approval_id"] == "approval_surface_1"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_surface_writer_rejects_unapproved_dry_run_payload() -> None:
    connection, surfaces = await _surface_runtime()
    try:
        generator, _provider = _generator({"surfaces": [_surface()]})
        report = await generator.generate([_memory()])
        writer = RetrievalSurfaceWriter(surfaces, CLOCK)

        with pytest.raises(ValidationError):
            await writer.write_approved(
                [report.surfaces[0].model_dump()],
                enable_write=True,
            )
        assert await surfaces.list_surfaces_for_memory(
            user_id="usr_1",
            memory_id="mem_1",
        ) == []
    finally:
        await connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_update",
    [
        {
            "surface_type": "alias",
            "anchor_type": None,
            "alias_kind": None,
            "preserve_verbatim": False,
        },
        {
            "surface_type": "anchor",
            "anchor_type": None,
            "alias_kind": None,
            "preserve_verbatim": False,
        },
        {
            "surface_type": "alias",
            "anchor_type": None,
            "alias_kind": "translation",
            "preserve_verbatim": True,
        },
    ],
)
async def test_surface_writer_rejects_semantically_invalid_approved_payloads(
    invalid_update: dict[str, Any],
) -> None:
    connection, surfaces = await _surface_runtime()
    try:
        approved = await _approved_surface()
        payload = approved.model_dump()
        payload.update(invalid_update)
        writer = RetrievalSurfaceWriter(surfaces, CLOCK)

        with pytest.raises(ValidationError):
            await writer.write_approved([payload], enable_write=True)
        assert await surfaces.list_surfaces_for_memory(
            user_id="usr_1",
            memory_id="mem_1",
        ) == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_surface_writer_rejects_wrong_user_memory_pair() -> None:
    connection, surfaces = await _surface_runtime()
    try:
        approved = (await _approved_surface()).model_copy(update={"user_id": "usr_2"})
        writer = RetrievalSurfaceWriter(surfaces, CLOCK)

        with pytest.raises(ValueError, match="memory_id must belong to user_id"):
            await writer.write_approved([approved], enable_write=True)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_surface_writer_rerun_is_idempotent() -> None:
    connection, surfaces = await _surface_runtime()
    try:
        approved = await _approved_surface()
        writer = RetrievalSurfaceWriter(surfaces, CLOCK)

        first_report = await writer.write_approved([approved], enable_write=True)
        second_report = await writer.write_approved([approved], enable_write=True)

        rows = await surfaces.list_surfaces_for_memory(
            user_id="usr_1",
            memory_id="mem_1",
        )
        assert len(rows) == 1
        assert first_report.results[0].surface_id == second_report.results[0].surface_id
        assert rows[0]["id"] == first_report.results[0].surface_id
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_synthetic_e2e_dry_run_writer_retrieval_and_composition() -> None:
    connection, surfaces = await _surface_runtime()
    try:
        generator, _provider = _generator(
            {
                "surfaces": [
                    _surface(
                        surface_text="direccion apartamento nuevo",
                        alias_kind="translation",
                        language_code="es",
                        derivation={"reason": "Spanish query bridge"},
                    )
                ]
            }
        )
        dry_run_report = await generator.generate([_memory()])
        approved = RetrievalSurfaceApprovedWrite.from_reviewed_surface(
            dry_run_report.surfaces[0],
            approval_id="approval_e2e_surface",
            approved_at=CLOCK.now().isoformat(),
            approved_by="synthetic-review",
        )
        writer = RetrievalSurfaceWriter(surfaces, CLOCK)
        search = CandidateSearch(connection, CLOCK)
        plan = _persisted_surface_plan("direccion apartamento nuevo")

        disabled_report = await writer.write_approved([approved])
        assert disabled_report.writes_enabled is False
        assert disabled_report.written_surface_count == 0
        assert await search.search(plan, user_id="usr_1", fts_query_audit=[]) == []

        write_report = await writer.write_approved([approved], enable_write=True)
        assert write_report.writes_enabled is True
        assert write_report.written_surface_count == 1

        fts_query_audit: list[dict[str, object]] = []
        candidates = await search.search(
            plan,
            user_id="usr_1",
            fts_query_audit=fts_query_audit,
        )

        assert [candidate["id"] for candidate in candidates] == ["mem_1"]
        assert candidates[0]["canonical_text"] == (
            "Ben's new apartment address is 120 Maple Street."
        )
        assert "direccion" not in candidates[0]["canonical_text"].lower()
        assert candidates[0]["fts_query_matches"][0]["source"] == "persisted_surface"
        assert candidates[0]["fts_query_matches"][0]["non_evidential"] is True
        persisted_surface_audit = [
            entry
            for entry in fts_query_audit
            if entry.get("source") == "persisted_surface"
        ]
        assert persisted_surface_audit[0]["raw_rows"] == 1

        composed = ContextComposer(CLOCK).compose(
            [
                ScoredCandidate(
                    memory_id=str(candidates[0]["id"]),
                    memory_object=candidates[0],
                    llm_applicability=1.0,
                    retrieval_score=float(candidates[0].get("rrf_score", 0.0)),
                    vitality_boost=0.0,
                    confirmation_boost=0.0,
                    need_boost=0.0,
                    penalty=0.0,
                    final_score=1.0,
                )
            ],
            current_contract={},
            user_state=None,
            resolved_policy=_resolved_policy(),
            conversation_messages=[],
            query_text="direccion apartamento nuevo",
            query_type="slot_fill",
            exact_recall_mode=True,
        )

        assert composed.selected_memory_ids == ["mem_1"]
        assert "Ben's new apartment address is 120 Maple Street." in composed.memory_block
        assert "direccion apartamento nuevo" not in composed.memory_block.lower()
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_synthetic_e2e_high_risk_surface_does_not_recover_active_candidate() -> None:
    connection, surfaces = await _surface_runtime()
    try:
        memories = MemoryObjectRepository(connection, CLOCK)
        await memories.create_memory_object(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            object_type=MemoryObjectType.EVIDENCE,
            scope=MemoryScope.GLOBAL_USER,
            canonical_text="The deployment approval code is stored separately.",
            source_kind=MemorySourceKind.EXTRACTED,
            confidence=0.9,
            privacy_level=0,
            memory_id="mem_high_risk_code",
        )
        generator, _provider = _generator(
            {
                "surfaces": [
                    _surface(
                        memory_id="mem_high_risk_code",
                        surface_type="anchor",
                        surface_text="SA42",
                        anchor_type="code",
                        alias_kind=None,
                        language_code=None,
                        preserve_verbatim=True,
                        derivation={"reason": "Exact code anchor"},
                    )
                ]
            }
        )
        dry_run_report = await generator.generate(
            [
                _memory(
                    "mem_high_risk_code",
                    canonical_text="The deployment approval code is stored separately.",
                )
            ]
        )
        approved = RetrievalSurfaceApprovedWrite.from_reviewed_surface(
            dry_run_report.surfaces[0],
            approval_id="approval_high_risk_code",
            approved_at=CLOCK.now().isoformat(),
            approved_by="synthetic-review",
        )
        writer = RetrievalSurfaceWriter(surfaces, CLOCK)

        write_report = await writer.write_approved([approved], enable_write=True)
        assert write_report.written_surface_count == 1

        candidates = await CandidateSearch(connection, CLOCK).search(
            _persisted_surface_plan("SA42"),
            user_id="usr_1",
            fts_query_audit=[],
        )

        assert candidates == []
    finally:
        await connection.close()
