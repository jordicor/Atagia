from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.asyncio
async def test_open_webui_filter_inlet_and_outlet_propagate_sidecar_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module(
        "atagia_memory_filter_runtime",
        ROOT / "integrations/open-webui/atagia_memory_filter.py",
    )
    calls: list[dict] = []

    def fake_post_json_sync(
        base_url,
        path,
        api_key,
        user_id,
        conversation_id,
        platform_id,
        payload,
        timeout_seconds,
        extra_headers,
    ):
        calls.append(
            {
                "path": path,
                "payload": payload,
                "headers": extra_headers,
                "user_id": user_id,
                "conversation_id": conversation_id,
                "platform_id": platform_id,
            }
        )
        if path.endswith("/context"):
            return {
                "system_prompt": "Remember the user likes short answers.",
                "request_message_id": "stored-user-1",
            }
        return {"ok": True}

    monkeypatch.setattr(module, "_post_json_sync", fake_post_json_sync)
    filter_instance = module.Filter()
    filter_instance.valves.api_key = "service-key"
    filter_instance.valves.memory_privacy_mode = "trusted_private"
    body = {
        "chat_id": "chat/with space",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    inlet_body = await filter_instance.inlet(
        body,
        __user__={"id": "usr"},
        __metadata__={"chat_id": "chat/with space"},
    )

    assert calls[0]["path"].endswith("/context")
    assert calls[0]["payload"]["message_text"] == "Hello"
    assert calls[0]["payload"]["ingest_origin"] == "live_turn"
    assert calls[0]["payload"]["confirmation_strategy"] == "live_prompt_allowed"
    assert calls[0]["payload"]["memory_privacy_mode"] == "trusted_private"
    assert calls[0]["headers"]["X-Atagia-Message-Id"] == calls[0]["payload"]["message_id"]
    assert inlet_body["messages"][0]["role"] == "system"
    assert "ATAGIA MEMORY CONTEXT" in inlet_body["messages"][0]["content"]

    inlet_body["messages"].append({"role": "assistant", "content": "Hi."})
    await filter_instance.outlet(
        inlet_body,
        __user__={"id": "usr"},
        __metadata__={"chat_id": "chat/with space"},
    )

    assert calls[1]["path"].endswith("/responses")
    assert calls[1]["payload"]["text"] == "Hi."
    assert calls[1]["payload"]["source_seq"] > calls[0]["payload"]["source_seq"]
    assert calls[1]["headers"]["X-Atagia-Response-Message-Id"] == calls[1]["payload"]["message_id"]
    debug_state = filter_instance.debug_state(
        __user__={"id": "usr"},
        __metadata__={"chat_id": "chat/with space"},
        body=inlet_body,
    )
    assert debug_state["status"] == "response_stored"
    assert debug_state["request_message_id"] == "stored-user-1"


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
