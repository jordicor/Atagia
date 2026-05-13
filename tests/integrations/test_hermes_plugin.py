from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]


def test_hermes_memory_provider_prefetch_and_background_sync(
    monkeypatch,
) -> None:
    module = _load_module(
        "atagia_hermes_provider_runtime",
        ROOT / "integrations/hermes/plugins/memory/atagia/provider.py",
    )
    provider = module.AtagiaMemoryProvider(
        {
            "base_url": "http://atagia.test",
            "api_key": "service-key",
            "user_id": "usr",
            "conversation_id": "cnv",
            "memory_privacy_mode": "trusted_private",
        }
    )
    calls: list[dict] = []

    def fake_request_json(path, payload, extra_headers=None):
        calls.append({"path": path, "payload": payload, "headers": extra_headers or {}})
        if path.endswith("/context"):
            return {
                "system_prompt": "Remember Hermes context.",
                "request_message_id": "stored-user-1",
            }
        return {"ok": True}

    monkeypatch.setattr(provider, "_request_json", fake_request_json)

    context = provider.prefetch("Hello", session_id="cnv")

    assert context["system_prompt"] == "Remember Hermes context."
    assert calls[0]["payload"]["ingest_origin"] == "live_turn"
    assert calls[0]["payload"]["confirmation_strategy"] == "live_prompt_allowed"
    assert calls[0]["payload"]["memory_privacy_mode"] == "trusted_private"
    assert calls[0]["headers"]["X-Atagia-Message-Id"] == calls[0]["payload"]["message_id"]

    assert provider.sync_turn(
        {
            "conversation_id": "cnv",
            "user_message": "Hello",
            "assistant_response": "Hi.",
        }
    )
    module.wait_for_queue(provider)

    assert any(call["path"].endswith("/messages") for call in calls)
    assert any(call["path"].endswith("/responses") for call in calls)
    assert provider.status()["status"] == "turn_synced"

    provider.on_session_end(
        {
            "conversation_id": "cnv",
            "messages": [
                {"role": "user", "content": "Backfill user"},
                {"role": "assistant", "content": "Backfill assistant"},
            ],
        }
    )
    module.wait_for_queue(provider)
    assert any(call["payload"]["ingest_origin"] == "backfill" for call in calls)
    assert any(call["payload"]["confirmation_strategy"] == "admin_review_only" for call in calls)
    assert provider.on_memory_write({"text": "curated memory"}) is False
    provider.shutdown()


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
