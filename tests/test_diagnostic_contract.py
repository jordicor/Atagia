"""Frozen, synthetic diagnostic capture contract fixture."""

import json
from pathlib import Path

from atagia.core.source_references import SourceReferenceCatalog
from atagia.diagnostics.contract import SCHEMA_VERSION, sha256_hex
from atagia.services.llm_client import LLMCompletionRequest, LLMCompletionResponse
from atagia.services.providers.typesafe import _EvaluationResponse


def test_synthetic_capture_fixture_is_complete_and_replayable() -> None:
    root = Path(__file__).parent / "fixtures" / "diagnostic_capture_v1"
    manifest = json.loads((root / "manifest.json").read_bytes())
    event_bytes = (root / "events.jsonl").read_bytes()
    events = [json.loads(line) for line in event_bytes.splitlines()]
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["status"] == "complete"
    assert manifest["event_count"] == len(events) == 4
    assert manifest["events_sha256"] == sha256_hex(event_bytes)
    assert [event["seq"] for event in events] == [1, 2, 3, 4]
    assert [event["phase"] for event in events if event["kind"] == "provider_attempt"] == ["start", "end"]
    start, end = events[1], events[2]
    request = LLMCompletionRequest.model_validate(json.loads((root / "blobs" / start["data"]["request"]["sha256"]).read_bytes()))
    sent = json.loads((root / "blobs" / start["data"]["sent_payload"]["sha256"]).read_bytes())
    returned = LLMCompletionResponse.model_validate(json.loads((root / "blobs" / end["data"]["response"]["sha256"]).read_bytes()))
    raw = _EvaluationResponse.model_validate(json.loads((root / "blobs" / end["data"]["raw_response"]["sha256"]).read_bytes()))
    assert sent["state"] == [{"role": message.role, "content": message.content} for message in request.messages]
    assert sent["questions"] == {key: question.model_dump(mode="json") for key, question in request.choice_questions.items()}
    assert returned.choice_answers["place"].choice == raw.answers["place"].choice == "attic"

    for event in events:
        for field in ("source", "request", "sent_payload", "response"):
            ref = event["data"].get(field)
            if ref is None:
                continue
            content = (root / "blobs" / ref["sha256"]).read_bytes()
            assert len(content) == ref["size_bytes"]
            assert sha256_hex(content) == ref["sha256"]

    source_ref = events[0]["data"]["source"]
    source = (root / "blobs" / source_ref["sha256"]).read_text(encoding="utf-8")
    catalog = SourceReferenceCatalog(source)
    selected = events[-1]["data"]["selected_reference"]
    assert catalog.resolve(selected["start_ref"], selected["end_ref"]).quote(source) == selected["quote"]
