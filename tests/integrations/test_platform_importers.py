from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[2]
IDENTITY = {
    "host_installation_id": "install-01",
    "host_account_id": "account-01",
    "host_conversation_id": "host-chat-01",
}


class RecordingImportClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def ingest_message(self, **kwargs):
        self.calls.append(kwargs)
        return {"created": True}


def test_sillytavern_import_preserves_order_and_reports_invalid_rows(
    tmp_path: Path,
) -> None:
    module = _module("atagia_importers_silly")
    source = tmp_path / "chat.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "is_user": True,
                        "mes": "yes",
                        "send_date": "2026-01-01T00:00:00Z",
                    }
                ),
                json.dumps({"name": "Assistant", "mes": "must not infer role"}),
                json.dumps({"is_user": False, "mes": ""}),
                json.dumps({"is_user": False, "mes": "yes"}),
            ]
        ),
        encoding="utf-8",
    )
    client = RecordingImportClient()

    summary = module.import_sillytavern_jsonl(
        source,
        client=client,
        user_id="usr",
        conversation_id="cnv",
        memory_privacy_mode="trusted_private",
        **IDENTITY,
    )

    assert summary.as_dict() == {
        "source": "sillytavern_jsonl",
        "total_records": 4,
        "imported": 2,
        "failed": 0,
        "skipped": 2,
        "skipped_by_reason": {"missing_role": 1, "empty_content": 1},
        "warnings": [
            "record 2: missing role",
            "record 3: empty content skipped",
        ],
        "errors": [],
    }
    assert summary.imported + summary.failed + summary.skipped == summary.total_records
    assert [call["role"] for call in client.calls] == ["user", "assistant"]
    assert [call["source_seq"] for call in client.calls] == [1, 4]
    assert client.calls[0]["message_id"] != client.calls[1]["message_id"]
    assert client.calls[0]["platform_id"] == "sillytavern"
    assert client.calls[0]["memory_privacy_mode"] == "trusted_private"


def test_ingestion_failures_count_as_failed_and_totals_reconcile(
    tmp_path: Path,
) -> None:
    module = _module("atagia_importers_ingest_failure")
    source = tmp_path / "chat.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"is_user": True, "mes": "first"}),
                json.dumps({"name": "Assistant", "mes": "must not infer role"}),
                json.dumps({"is_user": False, "mes": "second"}),
                json.dumps({"is_user": True, "mes": "third"}),
            ]
        ),
        encoding="utf-8",
    )

    class FailingSecondIngestClient(RecordingImportClient):
        def ingest_message(self, **kwargs):
            if kwargs["source_seq"] == 3:
                raise RuntimeError("upstream unavailable")
            return super().ingest_message(**kwargs)

    client = FailingSecondIngestClient()

    summary = module.import_sillytavern_jsonl(
        source,
        client=client,
        user_id="usr",
        conversation_id="cnv",
        **IDENTITY,
    )

    assert summary.total_records == 4
    assert summary.imported == 2
    assert summary.failed == 1
    assert summary.skipped == 1
    assert summary.skipped_by_reason == {"missing_role": 1}
    assert summary.errors == [
        "source_seq 3: ingestion failed: upstream unavailable"
    ]
    assert summary.imported + summary.failed + summary.skipped == summary.total_records
    assert summary.as_dict()["failed"] == 1
    assert [call["source_seq"] for call in client.calls] == [1, 4]


def test_strict_import_validates_entire_batch_before_network() -> None:
    module = _module("atagia_importers_strict")
    client = RecordingImportClient()
    payload = {
        "messages": [
            {"role": "user", "content": "valid"},
            {"role": "tool", "content": "unsupported"},
            {"content": "missing role"},
            {"role": "assistant", "content": ""},
            "not an object",
        ]
    }

    with pytest.raises(module.ImportBatchValidationError) as exc_info:
        module.import_openclaw_session(
            payload,
            client=client,
            user_id="usr",
            conversation_id="cnv",
            strict=True,
            **IDENTITY,
        )

    summary = exc_info.value.summary
    assert client.calls == []
    assert summary.imported == 0
    assert summary.skipped_by_reason == {
        "unsupported_role": 1,
        "missing_role": 1,
        "empty_content": 1,
        "invalid_record": 1,
    }
    assert len(summary.errors) == 4


def test_non_strict_jsonl_reports_malformed_and_non_object_rows(
    tmp_path: Path,
) -> None:
    module = _module("atagia_importers_malformed_jsonl")
    source = tmp_path / "chat.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"is_user": True, "mes": "first"}),
                "{not-json",
                json.dumps(["not", "an", "object"]),
                json.dumps({"is_user": False, "mes": "last"}),
            ]
        ),
        encoding="utf-8",
    )
    client = RecordingImportClient()

    summary = module.import_sillytavern_jsonl(
        source,
        client=client,
        user_id="usr",
        conversation_id="cnv",
        **IDENTITY,
    )

    assert summary.total_records == 4
    assert summary.imported == 2
    assert summary.skipped_by_reason == {"invalid_json": 1, "invalid_record": 1}
    assert [call["source_seq"] for call in client.calls] == [1, 4]


def test_hermes_curated_memories_are_reported_and_never_ingested() -> None:
    module = _module("atagia_importers_hermes")
    client = RecordingImportClient()
    payload = {
        "messages": [
            {"role": "user", "content": "Hermes user"},
            {"role": "assistant", "content": "Hermes assistant"},
            {"content": "roleless transcript row"},
        ],
        "memories": [
            {"content": "curated fact"},
            {"role": "assistant", "content": "still curated, not transcript"},
        ],
    }

    summary = module.import_hermes_export(
        payload,
        client=client,
        user_id="usr",
        conversation_id="cnv-hermes",
        **IDENTITY,
    )

    assert [call["text"] for call in client.calls] == [
        "Hermes user",
        "Hermes assistant",
    ]
    assert summary.total_records == 5
    assert summary.imported == 2
    assert summary.skipped_by_reason == {
        "curated_memory_unsupported": 2,
        "missing_role": 1,
    }
    assert summary.skipped == 3


def test_lorebook_reporter_never_fabricates_user_messages(tmp_path: Path) -> None:
    module = _module("atagia_importers_lorebook")
    source = tmp_path / "lore.txt"
    source.write_text("fact one\n\nfact two", encoding="utf-8")
    client = RecordingImportClient()

    summary = module.import_sillytavern_lorebook_text(
        source,
        client=client,
        user_id="usr",
        conversation_id="cnv",
        **IDENTITY,
    )

    assert client.calls == []
    assert summary.imported == 0
    assert summary.skipped_by_reason == {"curated_memory_unsupported": 2}


def test_message_identity_is_text_independent_global_and_rerunnable() -> None:
    module = _module("atagia_importers_identity")
    base = {
        "integration_kind": "openclaw",
        "host_installation_id": "install",
        "host_account_id": "account",
        "user_id": "usr",
        "host_conversation_id": "chat",
        "source_namespace": "host_message",
        "host_message_id": "local-7",
        "role": "assistant",
        "generation_id": "generation-1",
    }
    original = module.canonical_external_message_id(**base)
    assert original == module.canonical_external_message_id(**base)

    for field, value in (
        ("integration_kind", "hermes"),
        ("host_installation_id", "install-2"),
        ("host_account_id", "account-2"),
        ("user_id", "usr-2"),
        ("host_conversation_id", "chat-2"),
        ("source_namespace", "backfill_message"),
        ("host_message_id", "local-8"),
        ("role", "user"),
        ("generation_id", "generation-2"),
    ):
        changed = dict(base)
        changed[field] = value
        assert module.canonical_external_message_id(**changed) != original

    # Text is deliberately absent from the API and cannot affect identity.
    first_client = RecordingImportClient()
    second_client = RecordingImportClient()
    payload_a = {"messages": [{"id": "m1", "role": "user", "content": "alpha"}]}
    payload_b = {"messages": [{"id": "m1", "role": "user", "content": "beta"}]}
    module.import_openclaw_session(
        payload_a,
        client=first_client,
        user_id="usr",
        conversation_id="cnv",
        **IDENTITY,
    )
    module.import_openclaw_session(
        payload_b,
        client=second_client,
        user_id="usr",
        conversation_id="cnv",
        **IDENTITY,
    )
    assert first_client.calls[0]["message_id"] == second_client.calls[0]["message_id"]


def test_sillytavern_live_mapping_reconciles_selected_backfill() -> None:
    module = _module("atagia_importers_reconcile")
    identity = {
        "integration_kind": "sillytavern",
        "host_installation_id": IDENTITY["host_installation_id"],
        "host_account_id": IDENTITY["host_account_id"],
        "user_id": "usr",
        "host_conversation_id": IDENTITY["host_conversation_id"],
        "source_namespace": "host_message",
        "host_message_id": "browser-message-9",
        "role": "assistant",
        "generation_id": "swipe-2",
    }
    mapped_id = module.canonical_external_message_id(**identity)
    record = {
        "is_user": False,
        "mes": "selected swipe",
        "extra": {
            "atagia_source_identity": {
                "host_message_id": "browser-message-9",
                "host_generation_id": "swipe-2",
                "atagia_message_id": mapped_id,
                "source_seq": None,
            }
        },
    }
    client = RecordingImportClient()

    summary = module.import_sillytavern_jsonl(
        json.dumps(record),
        client=client,
        user_id="usr",
        conversation_id="cnv",
        **IDENTITY,
    )

    assert summary.imported == 1
    assert client.calls[0]["message_id"] == mapped_id
    assert client.calls[0]["source_seq"] is None

    bad_record = json.loads(json.dumps(record))
    bad_record["extra"]["atagia_source_identity"]["atagia_message_id"] = "other"
    bad_client = RecordingImportClient()
    bad_summary = module.import_sillytavern_jsonl(
        json.dumps(bad_record),
        client=bad_client,
        user_id="usr",
        conversation_id="cnv",
        **IDENTITY,
    )
    assert bad_client.calls == []
    assert bad_summary.skipped_by_reason == {"source_mapping_mismatch": 1}


def test_sillytavern_server_and_importer_share_the_canonical_identity_contract() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    module = _module("atagia_importers_cross_language")
    fields = {
        "integration_kind": "sillytavern",
        "host_installation_id": "installation",
        "host_account_id": "alice",
        "user_id": "atagia-alice",
        "host_conversation_id": "host/chat",
        "source_namespace": "host_message",
        "host_message_id": "message-7",
        "role": "assistant",
        "generation_id": "host:generation-2",
    }
    expected = module.canonical_external_message_id(**fields)
    plugin_path = ROOT / "integrations/sillytavern/server-plugin/index.cjs"
    script = (
        "const plugin=require(process.argv[1]);"
        "const fields=JSON.parse(process.argv[2]);"
        "process.stdout.write(plugin.canonicalExternalMessageId(fields));"
    )
    server_fields = {
        "installationId": fields["host_installation_id"],
        "hostAccountId": fields["host_account_id"],
        "mappedUserId": fields["user_id"],
        "hostConversationId": fields["host_conversation_id"],
        "sourceNamespace": fields["source_namespace"],
        "hostMessageId": fields["host_message_id"],
        "role": fields["role"],
        "generationId": fields["generation_id"],
    }
    result = subprocess.run(
        [node, "-e", script, str(plugin_path), json.dumps(server_fields)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout == expected


def test_backfill_retry_keeps_ids_and_monotonic_source_order() -> None:
    module = _module("atagia_importers_retry")
    payload = {
        "sessionFile": {
            "messages": [
                {"role": "user", "content": "yes"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "yes"},
            ]
        }
    }
    first = RecordingImportClient()
    retry = RecordingImportClient()
    kwargs = dict(
        user_id="usr",
        conversation_id="cnv-openclaw",
        **IDENTITY,
    )

    module.import_openclaw_session(payload, client=first, **kwargs)
    module.import_openclaw_session(payload, client=retry, **kwargs)

    assert [call["message_id"] for call in first.calls] == [
        call["message_id"] for call in retry.calls
    ]
    assert first.calls[0]["message_id"] != first.calls[2]["message_id"]
    assert [call["source_seq"] for call in first.calls] == [1, 2, 3]


def _module(name: str) -> ModuleType:
    return _load_module(name, ROOT / "integrations/importers/atagia_importers.py")


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module
