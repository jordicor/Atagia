"""Opt-in capture preserves the normal client result and provider call count."""

import json

import httpx
import pytest
from pydantic import BaseModel

from atagia.diagnostics.recorder import DiagnosticRecorder
from atagia.services.llm_client import LLMClient, LLMCompletionRequest, LLMCompletionResponse, LLMEmbeddingRequest, LLMEmbeddingResponse, LLMEmbeddingVector, LLMMessage, LLMProvider, LLMStreamEvent, RetryPolicy
from atagia.services.context_cache_service import ContextCacheService
from atagia.models.schemas_decisions import ChoiceQuestion
from atagia.services.providers.typesafe import TypeSafeProvider
from atagia.services.llm_client import LLMError
from tests.services.test_context_cache_service import _build_runtime, _seed_conversation


class ControlledProvider(LLMProvider):
    name = "openai"

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        self.calls += 1
        return LLMCompletionResponse(provider=self.name, model=request.model, output_text="done", usage={"input_tokens": 3, "output_tokens": 1})

    async def stream(self, request: LLMCompletionRequest):
        self.calls += 1
        yield LLMStreamEvent(type="text", content="partial")
        yield LLMStreamEvent(type="text", content=" rest")


async def test_capture_on_off_keeps_completion_and_call_count(tmp_path) -> None:
    request = LLMCompletionRequest(model="openai/test", messages=[LLMMessage(role="user", content="synthetic private input")], metadata={"purpose": "memory_extraction_candidate_card"})
    provider_off = ControlledProvider()
    client_off = LLMClient(providers=[provider_off], retry_policy=RetryPolicy(attempts=1))
    expected = await client_off.complete(request)

    recorder = DiagnosticRecorder(tmp_path)
    provider_on = ControlledProvider()
    client_on = LLMClient(providers=[provider_on], retry_policy=RetryPolicy(attempts=1), diagnostic_recorder=recorder)
    actual = await client_on.complete(request)
    recorder.close()

    assert actual == expected
    assert provider_off.calls == provider_on.calls == 1
    manifest = json.loads((recorder.root / "manifest.json").read_bytes())
    events = [json.loads(line) for line in (recorder.root / "events.jsonl").read_bytes().splitlines()]
    assert manifest["status"] == "complete"
    assert [event["kind"] for event in events] == ["operation_start", "provider_attempt", "provider_attempt", "operation_end"]
    assert [event["phase"] for event in events if event["kind"] == "provider_attempt"] == ["start", "end"]
    assert events[2]["data"]["cost_usd"] is None
    assert events[2]["data"]["cost_provenance"] == "unknown"
    assert events[2]["data"]["raw_response"] is not None
    assert (recorder.root / "blobs" / events[2]["data"]["raw_response"]["sha256"]).read_bytes() == b"{}\n"


async def test_capture_write_failure_does_not_repeat_successful_provider_call(tmp_path) -> None:
    recorder = DiagnosticRecorder(tmp_path, max_blob_bytes=8)
    provider = ControlledProvider()
    client = LLMClient(providers=[provider], retry_policy=RetryPolicy(attempts=3), diagnostic_recorder=recorder)
    response = await client.complete(LLMCompletionRequest(model="openai/test", messages=[LLMMessage(role="user", content="larger than eight bytes")]))
    assert response.output_text == "done"
    assert provider.calls == 1
    assert recorder.failed
    assert json.loads((recorder.root / "manifest.json").read_bytes())["status"] == "failed"


async def test_capture_serialization_failure_after_provider_success_is_latched(tmp_path) -> None:
    class OpaqueProvider(ControlledProvider):
        async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
            result = await super().complete(request)
            result.raw_response = {"opaque": object()}
            return result

    recorder = DiagnosticRecorder(tmp_path)
    provider = OpaqueProvider()
    client = LLMClient(providers=[provider], retry_policy=RetryPolicy(attempts=3), diagnostic_recorder=recorder)
    result = await client.complete(LLMCompletionRequest(model="openai/test", messages=[LLMMessage(role="user", content="synthetic")]))
    assert result.output_text == "done"
    assert provider.calls == 1
    assert recorder.failed


async def test_embedding_capture_records_shape_without_vector_values(tmp_path) -> None:
    class EmbeddingProvider(ControlledProvider):
        async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
            self.calls += 1
            return LLMEmbeddingResponse(provider=self.name, model=request.model, vectors=[LLMEmbeddingVector(index=0, values=[0.1234567, 0.7654321])])

    recorder = DiagnosticRecorder(tmp_path)
    provider = EmbeddingProvider()
    client = LLMClient(providers=[provider], diagnostic_recorder=recorder)
    response = await client.embed(LLMEmbeddingRequest(model="openai/embedding-test", input_texts=["synthetic text"]))
    recorder.close()
    assert response.vectors[0].values == [0.1234567, 0.7654321]
    assert provider.calls == 1
    events = [json.loads(line) for line in (recorder.root / "events.jsonl").read_bytes().splitlines()]
    start = next(event for event in events if event["kind"] == "provider_attempt" and event.get("phase") == "start")
    end = next(event for event in events if event["kind"] == "provider_attempt" and event.get("phase") == "end")
    assert start["data"]["request_kind"] == "embedding"
    summary = json.loads((recorder.root / "blobs" / end["data"]["response"]["sha256"]).read_bytes())
    assert summary["vector_count"] == 1 and summary["dimensions"] == [2]
    assert "values" not in summary


async def test_cancelled_stream_keeps_partial_text_and_one_attempt(tmp_path) -> None:
    recorder = DiagnosticRecorder(tmp_path)
    provider = ControlledProvider()
    client = LLMClient(providers=[provider], diagnostic_recorder=recorder)
    stream = client.stream(LLMCompletionRequest(model="openai/test", messages=[LLMMessage(role="user", content="synthetic")]))
    first = await anext(stream)
    assert first.content == "partial"
    await stream.aclose()
    recorder.close()
    events = [json.loads(line) for line in (recorder.root / "events.jsonl").read_bytes().splitlines()]
    terminal = [event for event in events if event["kind"] == "provider_attempt" and event.get("phase") == "end"]
    assert provider.calls == 1
    assert len(terminal) == 1 and terminal[0]["status"] == "cancelled"
    partial = terminal[0]["data"]["partial_output"]
    assert (recorder.root / "blobs" / partial["sha256"]).read_text(encoding="utf-8") == "partial"


async def test_typesafe_invalid_distribution_retains_sent_and_raw_payload(tmp_path) -> None:
    recorder = DiagnosticRecorder(tmp_path)
    response = {"model": "jev-test", "answers": {"decision": {"type": "choice", "choice": "yes", "probabilities": {"yes": 0.2, "no": 0.8}, "confidence": 0.2}}, "usage": {"input_tokens": 8, "output_tokens": 0}}
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response))) as http:
        client = LLMClient(providers=[TypeSafeProvider("synthetic-key", client=http)], diagnostic_recorder=recorder)
        request = LLMCompletionRequest(model="typesafe/jev-test", messages=[LLMMessage(role="user", content="synthetic state")], choice_questions={"decision": ChoiceQuestion(instructions="Choose.", criteria={"yes": None, "no": None})})
        with pytest.raises(LLMError, match="distribution"):
            await client.complete(request)
    recorder.close()
    events = [json.loads(line) for line in (recorder.root / "events.jsonl").read_bytes().splitlines()]
    payload = next(event for event in events if event["kind"] == "provider_payload")["data"]["sent_payload"]
    raw = next(event for event in events if event["kind"] == "provider_raw")["data"]["raw_response"]
    assert json.loads((recorder.root / "blobs" / payload["sha256"]).read_bytes())["questions"]["decision"]["criteria"] == {"yes": None, "no": None}
    assert json.loads((recorder.root / "blobs" / raw["sha256"]).read_bytes())["answers"]["decision"]["probabilities"]["yes"] == 0.2
    assert "synthetic-key" not in (recorder.root / "events.jsonl").read_text(encoding="utf-8")


async def test_structured_repair_correlates_validation_and_two_provider_calls(tmp_path) -> None:
    class ValueSchema(BaseModel):
        value: int

    class RepairProvider(ControlledProvider):
        async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
            self.calls += 1
            return LLMCompletionResponse(provider=self.name, model=request.model, output_text="invalid-json" if self.calls == 1 else '{"value":7}')

    request = LLMCompletionRequest(model="openai/test", messages=[LLMMessage(role="user", content="Return a value")], response_schema=ValueSchema.model_json_schema(), metadata={"purpose": "memory_extraction"})
    off_provider = RepairProvider()
    off = await LLMClient(providers=[off_provider], structured_output_retry_attempts=1).complete_structured_with_response(request, ValueSchema)
    recorder = DiagnosticRecorder(tmp_path)
    on_provider = RepairProvider()
    on = await LLMClient(providers=[on_provider], structured_output_retry_attempts=1, diagnostic_recorder=recorder).complete_structured_with_response(request, ValueSchema)
    recorder.close()
    assert on.value == off.value == ValueSchema(value=7)
    assert on_provider.calls == off_provider.calls == 2
    events = [json.loads(line) for line in (recorder.root / "events.jsonl").read_bytes().splitlines()]
    parent = next(event for event in events if event["kind"] == "operation_start" and event["purpose"] == "structured_completion")
    children = [event for event in events if event["kind"] == "operation_start" and event["purpose"] == "memory_extraction"]
    assert len(children) == 2 and all(event["parent_operation_id"] == parent["operation_id"] for event in children)
    validations = [event for event in events if event["kind"] == "no_call" and event["purpose"] == "structured_validation"]
    assert {event["data"]["status"] for event in validations} == {"failure", "success"}
    assert all(event["parent_operation_id"] == parent["operation_id"] for event in validations)


@pytest.mark.asyncio
async def test_real_sqlite_cache_reuse_on_off_has_same_calls(tmp_path, monkeypatch) -> None:
    async def run(enabled: bool):
        case_dir = tmp_path / ("on" if enabled else "off")
        case_dir.mkdir()
        runtime, provider = await _build_runtime(case_dir, monkeypatch)
        recorder = DiagnosticRecorder(tmp_path) if enabled else None
        if recorder is not None:
            runtime.llm_client._diagnostic_recorder = recorder
        try:
            await _seed_conversation(runtime, user_id="usr_1", conversation_id="cnv_1")
            service = ContextCacheService(runtime)
            connection = await runtime.open_connection()
            try:
                first = await service.resolve_with_connection(connection, user_id="usr_1", conversation_id="cnv_1", message_text="Please help me debug this retry loop.")
            finally:
                await connection.close()
            await service.publish_pending_cache_entry(first, last_retrieval_message_seq=1)
            connection = await runtime.open_connection()
            try:
                second = await service.resolve_with_connection(connection, user_id="usr_1", conversation_id="cnv_1", message_text="continue")
            finally:
                await connection.close()
            return second.from_cache, second.need_detection_skipped, [request.metadata.get("purpose") for request in provider.requests], recorder
        finally:
            await runtime.close()

    off = await run(False)
    on = await run(True)
    assert off[:3] == on[:3]
    assert on[0] is True and on[1] is True
    events = [json.loads(line) for line in (on[3].root / "events.jsonl").read_bytes().splitlines()]
    assert any(event["kind"] == "no_call" and event["purpose"] == "context_cache_reuse" for event in events)
