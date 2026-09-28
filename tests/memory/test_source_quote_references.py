"""Exercise selected source references through real cards and SQLite storage."""

from __future__ import annotations

import json

import pytest

from atagia.core.source_references import SourceReferenceCatalog
from atagia.services.llm_client import LLMCompletionRequest, LLMCompletionResponse
from tests.memory.test_extractor import (
    CannedExtractionProvider,
    _build_runtime_with_provider,
    _context,
    _create_source_message,
    _settings,
)


async def _source_span(connection) -> dict:
    row = await (
        await connection.execute(
            "SELECT * FROM memory_evidence_spans WHERE user_id = ? AND span_role = ?",
            ("usr_1", "source"),
        )
    ).fetchone()
    result = dict(row)
    result["metadata_json"] = json.loads(result["metadata_json"])
    return result


class ReferencedEvidenceProvider(CannedExtractionProvider):
    def __init__(self, reference_output: str, *, skip_first_chunk: bool = False, support_output: str | None = None) -> None:
        super().__init__(
            {
                "evidences": [
                    {
                        "canonical_text": "Use BLUE.",
                        "scope": "user",
                        "confidence": 0.9,
                    }
                ]
            }
        )
        self.reference_output = reference_output
        self.support_output = support_output
        self.skip_first_chunk = skip_first_chunk
        self.candidate_calls = 0

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        purpose = request.metadata.get("purpose")
        if purpose == "memory_extraction_candidate_card":
            self.candidate_calls += 1
            if self.skip_first_chunk and self.candidate_calls == 1:
                self.requests.append(request)
                return LLMCompletionResponse(
                    provider=self.name, model=request.model, output_text="none"
                )
        if purpose == "memory_extraction_evidence_support_card" and self.support_output is not None:
            self.requests.append(request)
            return LLMCompletionResponse(
                provider=self.name, model=request.model, output_text=self.support_output
            )
        if purpose == "memory_extraction_source_reference_card":
            self.requests.append(request)
            assert "first and last visible reference IDs" in request.messages[-1].content
            assert "[r1]" in request.messages[-1].content
            return LLMCompletionResponse(
                provider=self.name,
                model=request.model,
                output_text=self.reference_output,
            )
        return await super().complete(request)


@pytest.mark.asyncio
async def test_exact_second_occurrence_reaches_persisted_packet() -> None:
    source = "  Earlier: use BLUE.\r\nNow: use BLUE.\tKeep cafe\u0301.  "
    selected = "use BLUE.\tKeep cafe\u0301."
    provider = ReferencedEvidenceProvider("r8 r13")
    (
        connection,
        _clock,
        messages,
        _memories,
        extractor,
        _provider,
        policy,
    ) = await _build_runtime_with_provider(provider)
    try:
        await _create_source_message(messages, text=source)
        _result, persisted = await extractor.extract_with_persistence_details(
            message_text=source,
            role="user",
            conversation_context=_context("msg_1"),
            resolved_policy=policy,
        )
        assert len(persisted) == 1
        span = await _source_span(connection)
        assert span["quote_text"] == selected
        assert span["char_start"] == source.rindex("use BLUE.")
        assert span["char_end"] == len(source) - 2
        assert span["message_id"] == "msg_1"
        assert "quote_fallback" not in span["metadata_json"]
        assert (
            sum(
                request.metadata.get("purpose") == "memory_extraction_source_reference_card"
                for request in provider.requests
            )
            == 1
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wire",
    [
        "use BLUE",
        "r99 r100",
        "r3 r1",
        "r1 r2 r3",
        "r1 r3\nr1 r2",
        "",
    ],
)
async def test_invalid_evidence_stops_before_memory_is_written(wire: str) -> None:
    provider = ReferencedEvidenceProvider(wire)
    (
        connection,
        _clock,
        messages,
        _memories,
        extractor,
        _provider,
        policy,
    ) = await _build_runtime_with_provider(provider)
    try:
        await _create_source_message(messages, text="Use BLUE.")
        with pytest.raises(ValueError):
            await extractor.extract_with_persistence_details(
                message_text="Use BLUE.",
                role="user",
                conversation_context=_context("msg_1"),
                resolved_policy=policy,
            )
        row = await (
            await connection.execute("SELECT COUNT(*) AS count FROM memory_objects")
        ).fetchone()
        assert row["count"] == 0
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_no_support_is_an_explicit_selection_not_a_full_message_quote() -> None:
    provider = ReferencedEvidenceProvider("none", support_output="none")
    (
        connection,
        _clock,
        messages,
        memories,
        extractor,
        _provider,
        policy,
    ) = await _build_runtime_with_provider(provider)
    try:
        await _create_source_message(messages, text="Use BLUE.")
        result, persisted = await extractor.extract_with_persistence_details(
            message_text="Use BLUE.",
            role="user",
            conversation_context=_context("msg_1"),
            resolved_policy=policy,
        )
        assert result.nothing_durable and not persisted
        assert await memories.list_for_user("usr_1") == []
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_typesafe_reference_path_persists_metadata_and_literal_source(
    monkeypatch,
) -> None:
    source = "The shipment is delayed until Thursday."
    catalog = SourceReferenceCatalog(source)
    reference = catalog.resolve(
        catalog.anchors[0].reference_id,
        catalog.anchors[-1].reference_id,
    )

    class MetadataProvider(CannedExtractionProvider):
        async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
            if request.metadata.get("purpose") == "memory_extraction_evidence_support_card":
                self.requests.append(request)
                return LLMCompletionResponse(
                    provider=self.name,
                    model=request.model,
                    output_text="inferred",
                )
            return await super().complete(request)

    provider = MetadataProvider(
        {"evidences": [{
            "canonical_text": "The shipment is delayed until Thursday.",
            "scope": "user",
            "confidence": 0.9,
            "preserve_verbatim": True,
        }]}
    )

    async def select(_client, **kwargs):
        assert kwargs["source_text"] == source
        assert "<recent_context>" in kwargs["source_context"]
        return {"cand_001": reference}

    monkeypatch.setattr(
        "atagia.memory.source_quote_selector.select_source_references", select
    )
    settings = _settings(
        llm_finite_decisions_enabled=True,
        llm_component_models={"extraction_evidence": "typesafe/jev-latest"}
    )
    (
        connection,
        _clock,
        messages,
        memories,
        extractor,
        _provider,
        policy,
    ) = await _build_runtime_with_provider(provider, settings=settings)
    try:
        await _create_source_message(messages, text=source)
        result, persisted = await extractor.extract_with_persistence_details(
            message_text=source,
            role="user",
            conversation_context=_context("msg_1"),
            resolved_policy=policy,
        )
        assert len(result.evidences) == len(persisted) == 1
        assert result.evidences[0].source_quote == source
        assert result.evidences[0].preserve_verbatim is True
        assert result.evidences[0].language_codes == ["en"]
        span = await _source_span(connection)
        assert span["quote_text"] == source
        edge = await (
            await connection.execute(
                "SELECT support_kind FROM memory_support_edges WHERE user_id = ?",
                ("usr_1",),
            )
        ).fetchone()
        assert edge["support_kind"] == "inferred"
        assert len(await memories.list_for_user("usr_1")) == 1
        assert sum(
            request.metadata.get("purpose") == "memory_extraction_evidence_support_card"
            for request in provider.requests
        ) == 1
        assert not any(
            request.metadata.get("purpose") == "memory_extraction_source_reference_card"
            for request in provider.requests
        )
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_different_stored_source_is_rejected_before_model_dispatch() -> None:
    provider = ReferencedEvidenceProvider("r1 r3")
    (
        connection,
        _clock,
        messages,
        _memories,
        extractor,
        _provider,
        policy,
    ) = await _build_runtime_with_provider(provider)
    try:
        await _create_source_message(messages, text="Use RED.")
        with pytest.raises(ValueError, match="stored source message exactly"):
            await extractor.extract_with_persistence_details(
                message_text="Use BLUE.",
                role="user",
                conversation_context=_context("msg_1"),
                resolved_policy=policy,
            )
        assert not provider.requests
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_second_repeated_chunk_keeps_absolute_source_coordinates() -> None:
    paragraph = "Use BLUE.\r\n" + "background " * 120
    source = f"  {paragraph}\r\n\r\n{paragraph}  "
    provider = ReferencedEvidenceProvider(
        "r1 r3", skip_first_chunk=True
    )
    settings = _settings(chunking_extraction_threshold_tokens=20)
    (
        connection,
        _clock,
        messages,
        _memories,
        extractor,
        _provider,
        policy,
    ) = await _build_runtime_with_provider(provider, settings=settings)
    try:
        await _create_source_message(messages, text=source)
        _result, persisted = await extractor.extract_with_persistence_details(
            message_text=source,
            role="user",
            conversation_context=_context("msg_1"),
            resolved_policy=policy,
        )
        assert provider.candidate_calls == 2
        assert len(persisted) == 1
        span = await _source_span(connection)
        assert span["quote_text"] == "Use BLUE."
        assert span["char_start"] == source.rindex("Use BLUE.")
        assert span["char_end"] == span["char_start"] + len("Use BLUE.")
    finally:
        await connection.close()
