"""Regenerate the synthetic version 1 diagnostic capture fixture."""

from __future__ import annotations

import json
from pathlib import Path

from atagia.diagnostics.contract import SCHEMA_VERSION, canonical_json_bytes, sha256_hex
from atagia.core.source_references import SourceReferenceCatalog
from atagia.models.schemas_decisions import ChoiceQuestion
from atagia.services.llm_client import LLMCompletionRequest, LLMMessage
from atagia.services.llm_client import LLMCompletionResponse
from atagia.models.schemas_decisions import ChoiceAnswer

ROOT = Path(__file__).resolve().parent
CREATED = "2026-09-26T10:00:00Z"


def blob(value: object) -> dict[str, object]:
    data = value.encode("utf-8") if isinstance(value, str) else canonical_json_bytes(value)
    digest = sha256_hex(data)
    directory = ROOT / "blobs"
    directory.mkdir(exist_ok=True)
    (directory / digest).write_bytes(data)
    return {"sha256": digest, "size_bytes": len(data), "encoding": "utf-8"}


source = "Ada said the blue lantern is in the attic."
catalog = SourceReferenceCatalog(source)
start_ref = catalog.anchors[0].reference_id
end_ref = catalog.anchors[-1].reference_id
selected = catalog.resolve(start_ref, end_ref)
request = LLMCompletionRequest(model="jev-test", messages=[LLMMessage(role="user", content=source)], choice_questions={"place": ChoiceQuestion(instructions="Where is the lantern?", criteria={"attic": None, "cellar": None})}, metadata={"purpose": "memory_extraction_evidence_card"})
input_ref = blob(request.model_dump(mode="json"))
sent_ref = blob({"model": request.model, "state": [{"role": message.role, "content": message.content} for message in request.messages], "questions": {key: question.model_dump(mode="json") for key, question in request.choice_questions.items()}})
raw = {"model": "jev-test-revision", "answers": {"place": {"type": "choice", "choice": "attic", "probabilities": {"attic": 1.0, "cellar": 0.0}, "confidence": 1.0}}, "usage": {"input_tokens": 12, "output_tokens": 0}}
raw_ref = blob(raw)
response = LLMCompletionResponse(provider="typesafe", model="jev-test-revision", choice_answers={"place": ChoiceAnswer.model_validate(raw["answers"]["place"])}, usage=raw["usage"], finish_reason="stop", raw_response=raw)
output_ref = blob(response.model_dump(mode="json"))
source_ref = blob(source)
events = [
    {"seq": 1, "timestamp": CREATED, "kind": "operation_start", "trace_id": "synthetic-trace", "operation_id": "synthetic-card", "parent_operation_id": None, "attempt_id": None, "purpose": "memory_extraction_evidence_card", "component": "extractor", "card": "evidence", "user_id": "synthetic-user", "turn_id": None, "job_id": "synthetic-job", "status": "started", "data": {"source": source_ref, "source_catalog": {"version": 1, "source_sha256": catalog.source_hash, "anchors": [{"reference_id": a.reference_id, "char_start": a.char_start, "char_end": a.char_end} for a in catalog.anchors]}, "provenance": {"code_revision": "synthetic-revision", "prompt_builder": "synthetic_fixture", "prompt_sha256": input_ref["sha256"], "contract_sha256": sent_ref["sha256"]}}},
    {"seq": 2, "timestamp": CREATED, "kind": "provider_attempt", "phase": "start", "trace_id": "synthetic-trace", "operation_id": "synthetic-card", "parent_operation_id": None, "attempt_id": "synthetic-attempt", "purpose": "memory_extraction_evidence_card", "component": "extractor", "card": "evidence", "user_id": "synthetic-user", "turn_id": None, "job_id": "synthetic-job", "status": "started", "data": {"request": input_ref, "sent_payload": sent_ref, "requested_model": "typesafe/jev-test", "resolved_provider": "typesafe", "resolved_model": "jev-test", "parameters": {"temperature": 0.0, "max_output_tokens": 64}, "started_at": CREATED}},
    {"seq": 3, "timestamp": CREATED, "kind": "provider_attempt", "phase": "end", "trace_id": "synthetic-trace", "operation_id": "synthetic-card", "parent_operation_id": None, "attempt_id": "synthetic-attempt", "purpose": "memory_extraction_evidence_card", "component": "extractor", "card": "evidence", "user_id": "synthetic-user", "turn_id": None, "job_id": "synthetic-job", "status": "success", "data": {"response": output_ref, "raw_response": raw_ref, "parsed": {"place": "attic"}, "usage": {"input_tokens": 12, "output_tokens": 0}, "usage_provenance": "provider", "cost_usd": None, "cost_provenance": "unknown", "finished_at": CREATED}},
    {"seq": 4, "timestamp": CREATED, "kind": "operation_end", "trace_id": "synthetic-trace", "operation_id": "synthetic-card", "parent_operation_id": None, "attempt_id": None, "purpose": "memory_extraction_evidence_card", "component": "extractor", "card": "evidence", "user_id": "synthetic-user", "turn_id": None, "job_id": "synthetic-job", "status": "success", "data": {"selected_reference": {"start_ref": start_ref, "end_ref": end_ref, "quote": selected.quote(source)}, "memory_ids": ["synthetic-memory"]}},
]
event_bytes = b"".join(canonical_json_bytes(event) for event in events)
(ROOT / "events.jsonl").write_bytes(event_bytes)
manifest = {"schema_version": SCHEMA_VERSION, "capture_id": "synthetic-capture", "created_at": CREATED, "finished_at": CREATED, "status": "complete", "event_count": len(events), "events_sha256": sha256_hex(event_bytes), "limits": {"max_blob_bytes": 1048576, "max_session_bytes": 10485760}, "failure_reason": None}
(ROOT / "manifest.json").write_bytes(canonical_json_bytes(manifest))
referenced = {input_ref["sha256"], sent_ref["sha256"], output_ref["sha256"], raw_ref["sha256"], source_ref["sha256"]}
for path in (ROOT / "blobs").iterdir():
    if path.name not in referenced:
        path.unlink()
