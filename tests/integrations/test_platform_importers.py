from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]


class RecordingImportClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def ingest_message(self, **kwargs):
        self.calls.append(kwargs)
        return {"created": True}


def test_sillytavern_jsonl_importer_backfills_messages(tmp_path: Path) -> None:
    module = _load_module(
        "atagia_importers_runtime",
        ROOT / "integrations/importers/atagia_importers.py",
    )
    source = tmp_path / "chat.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"is_user": True, "mes": "Hello", "send_date": "2026-01-01T00:00:00Z"}),
                json.dumps({"is_user": False, "mes": "Hi"}),
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
    )

    assert summary.imported == 2
    assert [call["role"] for call in client.calls] == ["user", "assistant"]
    assert client.calls[0]["platform_id"] == "sillytavern"
    assert client.calls[0]["message_id"] == client.calls[0]["message_id"]
    assert client.calls[0]["source_seq"] > 0
    assert client.calls[0]["memory_privacy_mode"] == "trusted_private"


def test_openclaw_and_hermes_importers_are_rerunnable() -> None:
    module = _load_module(
        "atagia_importers_runtime_repeat",
        ROOT / "integrations/importers/atagia_importers.py",
    )
    openclaw_payload = {
        "sessionFile": {
            "messages": [
                {"role": "user", "content": "OpenClaw user"},
                {"role": "assistant", "content": "OpenClaw assistant"},
            ]
        }
    }
    hermes_payload = {
        "messages": [
            {"role": "user", "content": "Hermes user"},
            {"role": "assistant", "content": "Hermes assistant"},
        ]
    }
    first_client = RecordingImportClient()
    second_client = RecordingImportClient()

    module.import_openclaw_session(
        openclaw_payload,
        client=first_client,
        user_id="usr",
        conversation_id="cnv-openclaw",
    )
    module.import_openclaw_session(
        openclaw_payload,
        client=second_client,
        user_id="usr",
        conversation_id="cnv-openclaw",
    )

    assert [call["message_id"] for call in first_client.calls] == [
        call["message_id"] for call in second_client.calls
    ]
    assert all(call["platform_id"] == "openclaw" for call in first_client.calls)

    hermes_client = RecordingImportClient()
    summary = module.import_hermes_export(
        hermes_payload,
        client=hermes_client,
        user_id="usr",
        conversation_id="cnv-hermes",
    )

    assert summary.imported == 2
    assert all(call["platform_id"] == "hermes" for call in hermes_client.calls)


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
