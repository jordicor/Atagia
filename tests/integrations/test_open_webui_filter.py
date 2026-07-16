from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[2]


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _configured_filter(module, *, clock=None):
    instance = module.Filter(clock=clock)
    instance.valves.api_key = "service-key"
    instance.valves.installation_id = "owui-install-01"
    return instance


@pytest.mark.asyncio
async def test_v090_contract_runs_full_inlet_outlet_and_live_metadata_correlation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module("atagia_memory_filter_contract")
    runtime = _load_module(
        "open_webui_v090_runtime",
        ROOT / "tests/integrations/fixtures/open_webui_v0_9_0_filter_runtime.py",
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
    filter_instance = _configured_filter(module)
    filter_instance.valves.memory_privacy_mode = "trusted_private"
    metadata = {"chat_id": "host/chat 1", "atagia_conversation_id": "atagia/chat 1"}
    body = {
        "messages": [{"id": "host-user-1", "role": "user", "content": "Hello"}],
    }

    inlet_body = await runtime.call_filter_hook(
        filter_instance,
        "inlet",
        body,
        user={"id": "host-account-1"},
        metadata=metadata,
    )

    assert calls[0]["path"].endswith("/context")
    assert calls[0]["payload"]["message_text"] == "Hello"
    assert calls[0]["payload"]["ingest_origin"] == "live_turn"
    assert calls[0]["payload"]["confirmation_strategy"] == "live_prompt_allowed"
    assert calls[0]["payload"]["memory_privacy_mode"] == "trusted_private"
    assert (
        calls[0]["headers"]["X-Atagia-Message-Id"] == calls[0]["payload"]["message_id"]
    )
    assert inlet_body["messages"][0]["role"] == "system"
    assert "ATAGIA:FILTER:MEMORY_CONTEXT:v1" in inlet_body["messages"][0]["content"]
    correlation_token = metadata[module._CORRELATION_KEY]
    assert isinstance(correlation_token, str)
    assert correlation_token in filter_instance._correlations
    assert calls[0]["user_id"] == "host-account-1"

    inlet_body["messages"].append(
        {"id": "host-assistant-1", "role": "assistant", "content": "Hi."}
    )
    await runtime.call_filter_hook(
        filter_instance,
        "outlet",
        inlet_body,
        user={"id": "host-account-1"},
        metadata=metadata,
    )

    assert calls[1]["path"].endswith("/responses")
    assert calls[1]["payload"]["text"] == "Hi."
    assert calls[1]["payload"]["source_seq"] > calls[0]["payload"]["source_seq"]
    assert (
        calls[1]["headers"]["X-Atagia-Response-Message-Id"]
        == calls[1]["payload"]["message_id"]
    )
    assert module._CORRELATION_KEY not in metadata
    debug_state = filter_instance.debug_state(
        __user__={"id": "host-account-1"},
        __metadata__={
            "chat_id": "host/chat 1",
            "atagia_conversation_id": "atagia/chat 1",
        },
        body=inlet_body,
    )
    assert debug_state["status"] == "response_stored"
    assert debug_state["request_message_id"] == "stored-user-1"
    assert "preview" not in debug_state
    assert "Remember" not in repr(debug_state)


@pytest.mark.asyncio
async def test_reentry_replaces_only_owned_block_and_preserves_other_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module("atagia_memory_filter_replace")
    prompts = iter(["old private context", "new private context", "tool-loop context"])

    def fake_post(*args, **kwargs):
        return {"system_prompt": next(prompts), "request_message_id": "stored"}

    monkeypatch.setattr(module, "_post_json_sync", fake_post)
    instance = _configured_filter(module)
    unrelated = {
        "role": "system",
        "content": "Unrelated system bytes\nremain identical.",
        "custom": {"nested": [1, 2, 3]},
    }
    user_message = {"id": "u-1", "role": "user", "content": "same"}
    unrelated_before = copy.deepcopy(unrelated)
    user_before = copy.deepcopy(user_message)
    body = {"messages": [unrelated, user_message]}
    metadata = {"chat_id": "chat-1"}

    await instance.inlet(body, __user__={"id": "account-1"}, __metadata__=metadata)
    first_id = instance.debug_state(
        __user__={"id": "account-1"}, __metadata__=metadata
    )["request_message_id"]
    await instance.inlet(body, __user__={"id": "account-1"}, __metadata__=metadata)

    owned = [
        message for message in body["messages"] if module._is_owned_message(message)
    ]
    assert len(owned) == 1
    assert "new private context" in owned[0]["content"]
    assert "old private context" not in owned[0]["content"]
    assert unrelated == unrelated_before
    assert user_message == user_before
    assert (
        first_id
        == instance.debug_state(__user__={"id": "account-1"}, __metadata__=metadata)[
            "request_message_id"
        ]
    )

    # A tool continuation gets one current block as well.
    body["messages"].extend(
        [
            {"role": "assistant", "content": "calling tool"},
            {"role": "tool", "content": "tool output"},
            {"id": "u-2", "role": "user", "content": "same"},
        ]
    )
    await instance.inlet(body, __user__={"id": "account-1"}, __metadata__=metadata)
    owned = [
        message for message in body["messages"] if module._is_owned_message(message)
    ]
    assert len(owned) == 1
    assert "tool-loop context" in owned[0]["content"]


@pytest.mark.asyncio
async def test_empty_context_disable_and_fail_open_remove_stale_owned_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module("atagia_memory_filter_stale")
    instance = _configured_filter(module)
    body = {"messages": [{"id": "u1", "role": "user", "content": "hello"}]}
    metadata = {"chat_id": "chat"}
    responses = iter(
        [
            {"system_prompt": "first"},
            {"system_prompt": ""},
        ]
    )
    monkeypatch.setattr(module, "_post_json_sync", lambda *a, **k: next(responses))

    await instance.inlet(body, __user__={"id": "account"}, __metadata__=metadata)
    assert sum(module._is_owned_message(message) for message in body["messages"]) == 1
    await instance.inlet(body, __user__={"id": "account"}, __metadata__=metadata)
    assert not any(module._is_owned_message(message) for message in body["messages"])

    module._replace_owned_context(body["messages"], "stale")
    instance.valves.enabled = False
    await instance.inlet(body, __user__={"id": "account"}, __metadata__=metadata)
    assert not any(module._is_owned_message(message) for message in body["messages"])

    module._replace_owned_context(body["messages"], "stale again")
    instance.valves.enabled = True
    monkeypatch.setattr(
        module,
        "_post_json_sync",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("raw secret failure")),
    )
    emitted: list[dict] = []

    async def emitter(event):
        emitted.append(event)

    await instance.inlet(
        body,
        __user__={"id": "account"},
        __metadata__=metadata,
        __event_emitter__=emitter,
    )
    assert not any(module._is_owned_message(message) for message in body["messages"])
    state = instance.debug_state(__user__={"id": "account"}, __metadata__=metadata)
    assert state["error_code"] == "upstream_context_unavailable"
    assert "secret" not in repr(state)
    assert "secret" not in repr(emitted)


def test_canonical_identity_is_text_free_and_scoped_to_every_host_dimension() -> None:
    module = _module("atagia_memory_filter_identity")
    base = {
        "installation_id": "install",
        "host_account_id": "account",
        "user_id": "user",
        "host_conversation_id": "chat",
        "source_namespace": "host_message",
        "host_message_id": "message-1",
        "role": "assistant",
        "generation_id": "generation-1",
    }
    identity = module._canonical_message_id(**base)
    assert identity == module._canonical_message_id(**base)
    for field, changed_value in (
        ("installation_id", "install-2"),
        ("host_account_id", "account-2"),
        ("user_id", "user-2"),
        ("host_conversation_id", "chat-2"),
        ("source_namespace", "live_event"),
        ("host_message_id", "message-2"),
        ("role", "user"),
        ("generation_id", "generation-2"),
    ):
        changed = dict(base)
        changed[field] = changed_value
        assert module._canonical_message_id(**changed) != identity


def test_identity_fallbacks_are_not_shared_placeholders() -> None:
    module = _module("atagia_memory_filter_required_identity")
    instance = module.Filter()

    assert instance.valves.default_host_account_id == ""
    assert instance.valves.default_user_id == ""
    assert instance.valves.default_conversation_id == ""
    with pytest.raises(ValueError, match="Open WebUI account must be configured"):
        instance._resolve_scope(None, {}, prefer_correlation=False)


def test_single_user_mapping_is_bound_to_the_configured_host_account() -> None:
    module = _module("atagia_memory_filter_single_user_mapping")
    instance = module.Filter()
    instance.valves.default_host_account_id = "only-host"
    instance.valves.default_user_id = "mapped-atagia-user"

    scope = instance._resolve_scope(
        {"id": "only-host"},
        {"chat_id": "chat"},
        prefer_correlation=False,
    )
    assert scope.host_account_id == "only-host"
    assert scope.atagia_user_id == "mapped-atagia-user"
    with pytest.raises(ValueError, match="does not match __user__"):
        instance._resolve_scope(
            {"id": "different-host"},
            {"chat_id": "chat"},
            prefer_correlation=False,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "claimed"),
    (
        ("open_webui_account_id", "victim-host"),
        ("atagia_user_id", "victim-user"),
        ("user_id", "victim-user"),
    ),
)
async def test_inlet_rejects_body_metadata_identity_spoofing(
    field: str,
    claimed: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(f"atagia_memory_filter_inlet_spoof_{field}")
    calls: list[tuple] = []
    monkeypatch.setattr(
        module,
        "_post_json_sync",
        lambda *args, **_kwargs: calls.append(args) or {"system_prompt": "leak"},
    )
    instance = _configured_filter(module)
    body = {
        "metadata": {field: claimed},
        "messages": [{"id": "u-1", "role": "user", "content": "hello"}],
    }

    result = await instance.inlet(
        body,
        __user__={"id": "attacker"},
        __metadata__={"chat_id": "attacker-chat"},
    )

    assert result is body
    assert calls == []
    assert module._CORRELATION_KEY not in body["metadata"]


@pytest.mark.asyncio
async def test_outlet_rejects_forged_and_cross_user_correlation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module("atagia_memory_filter_outlet_spoof")
    calls: list[dict] = []

    def fake_post(*args, **_kwargs):
        calls.append(args[6])
        return {"system_prompt": "victim context", "request_message_id": "stored"}

    monkeypatch.setattr(module, "_post_json_sync", fake_post)
    instance = _configured_filter(module)
    victim_metadata = {"chat_id": "victim-chat"}
    victim_body = {"messages": [{"id": "v-u-1", "role": "user", "content": "victim"}]}
    await instance.inlet(
        victim_body,
        __user__={"id": "victim"},
        __metadata__=victim_metadata,
    )
    token = victim_metadata[module._CORRELATION_KEY]
    assert token in instance._correlations

    attacker_body = {
        "messages": [
            {"id": "a-u-1", "role": "user", "content": "attack"},
            {"id": "a-a-1", "role": "assistant", "content": "forged"},
        ]
    }
    attacker_metadata = {
        "chat_id": "attacker-chat",
        module._CORRELATION_KEY: token,
    }
    await instance.outlet(
        attacker_body,
        __user__={"id": "attacker"},
        __metadata__=attacker_metadata,
    )
    assert len(calls) == 1
    assert token in instance._correlations
    assert module._CORRELATION_KEY not in attacker_metadata

    forged_metadata = {
        "chat_id": "attacker-chat",
        module._CORRELATION_KEY: {
            "atagia_user_id": "victim",
            "atagia_conversation_id": "victim-chat",
            "host_account_id": "victim",
            "host_conversation_id": "victim-chat",
        },
    }
    await instance.outlet(
        attacker_body,
        __user__={"id": "attacker"},
        __metadata__=forged_metadata,
    )
    assert len(calls) == 1
    assert module._CORRELATION_KEY not in forged_metadata


@pytest.mark.asyncio
async def test_outlet_rejects_same_user_cross_conversation_correlation_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module("atagia_memory_filter_cross_conversation")
    calls: list[dict] = []

    def fake_post(*args, **_kwargs):
        calls.append(args[6])
        return {"system_prompt": "context", "request_message_id": "stored"}

    monkeypatch.setattr(module, "_post_json_sync", fake_post)
    instance = _configured_filter(module)
    issued_metadata = {"chat_id": "chat-a"}
    await instance.inlet(
        {"messages": [{"id": "u-1", "role": "user", "content": "hello"}]},
        __user__={"id": "owner"},
        __metadata__=issued_metadata,
    )
    token = issued_metadata[module._CORRELATION_KEY]
    assert len(calls) == 1

    replay_body = {
        "messages": [
            {"id": "u-2", "role": "user", "content": "other chat"},
            {"id": "a-2", "role": "assistant", "content": "replayed"},
        ]
    }
    replay_metadata = {
        "chat_id": "chat-b",
        module._CORRELATION_KEY: token,
    }
    await instance.outlet(
        replay_body,
        __user__={"id": "owner"},
        __metadata__=replay_metadata,
    )
    assert len(calls) == 1
    assert token in instance._correlations
    assert module._CORRELATION_KEY not in replay_metadata

    legitimate_metadata = {
        "chat_id": "chat-a",
        module._CORRELATION_KEY: token,
    }
    await instance.outlet(
        {
            "messages": [
                {"id": "u-1", "role": "user", "content": "hello"},
                {"id": "a-1", "role": "assistant", "content": "answer"},
            ]
        },
        __user__={"id": "owner"},
        __metadata__=legitimate_metadata,
    )
    assert len(calls) == 2
    assert token not in instance._correlations


@pytest.mark.asyncio
async def test_repeated_text_gets_distinct_ordinal_ids_and_retry_is_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module("atagia_memory_filter_repeated")
    calls: list[dict] = []

    def fake_post(*args, **kwargs):
        calls.append(args[6])
        return {"system_prompt": "context"}

    monkeypatch.setattr(module, "_post_json_sync", fake_post)
    instance = _configured_filter(module)
    metadata = {"chat_id": "chat"}
    body = {"messages": [{"role": "user", "content": "yes"}]}

    await instance.inlet(body, __user__={"id": "account"}, __metadata__=metadata)
    await instance.inlet(body, __user__={"id": "account"}, __metadata__=metadata)
    assert calls[0]["message_id"] == calls[1]["message_id"]
    assert calls[0]["source_seq"] == calls[1]["source_seq"]

    body["messages"].extend(
        [
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "yes"},
        ]
    )
    await instance.inlet(body, __user__={"id": "account"}, __metadata__=metadata)
    assert calls[2]["message_id"] != calls[0]["message_id"]
    assert calls[2]["source_seq"] > calls[0]["source_seq"]


def test_diagnostic_state_is_lru_ttl_bounded_and_lifecycle_purge_is_isolated() -> None:
    module = _module("atagia_memory_filter_state")
    clock = FakeClock()
    instance = _configured_filter(module, clock=clock)
    instance.valves.diagnostic_cache_max_entries = 128
    instance.valves.diagnostic_cache_ttl_seconds = 10

    for index in range(2500):
        scope = module._Scope(
            f"user-{index % 7}",
            f"chat-{index}",
            f"account-{index % 7}",
            f"host-{index}",
        )
        instance._state_set(scope, {"status": "context_empty", "has_context": False})
    assert len(instance._diagnostic_state) == 128
    assert ("user-0", "chat-0") not in instance._diagnostic_state

    target_a = module._Scope("isolated-user", "chat-a", "a", "ha")
    target_b = module._Scope("isolated-user", "chat-b", "a", "hb")
    other = module._Scope("other-user", "chat-a", "b", "ha")
    instance._state_set(target_a, {"status": "a"})
    instance._state_set(target_b, {"status": "b"})
    instance._state_set(other, {"status": "other"})
    target_a_token = instance._correlation_set(target_a)
    target_b_token = instance._correlation_set(target_b)
    other_token = instance._correlation_set(other)
    assert instance.delete_state(user_id="isolated-user", conversation_id="chat-a") == 1
    assert instance._state_get(target_a) is None
    assert instance._state_get(target_b) == {"status": "b"}
    assert instance._state_get(other) == {"status": "other"}
    assert instance._correlation_get(target_a_token) is None
    assert instance._correlation_get(target_b_token) is not None
    assert instance._correlation_get(other_token) is not None
    assert instance.delete_state(user_id="isolated-user") == 1
    assert instance._correlation_get(target_b_token) is None

    clock.advance(10)
    assert instance._state_get(other) is None
    assert instance._correlation_get(other_token) is None
    assert len(instance._diagnostic_state) == 0
    assert len(instance._correlations) == 0


def test_generation_without_host_sequence_omits_source_seq() -> None:
    module = _module("atagia_memory_filter_generation")
    scope = module._Scope("user", "cnv", "account", "host-chat")
    info = module._latest_message_info(
        [
            {"role": "user", "content": "prompt"},
            {
                "id": "assistant-message",
                "generation_id": "regeneration-2",
                "role": "assistant",
                "content": "new answer",
            },
        ],
        role="assistant",
        scope=scope,
        metadata={},
        installation_id="install",
    )
    assert info is not None
    assert info["source_seq"] is None

    explicit = module._latest_message_info(
        [
            {
                "id": "assistant-message",
                "generation_id": "regeneration-2",
                "atagia_source_seq": 9,
                "role": "assistant",
                "content": "new answer",
            }
        ],
        role="assistant",
        scope=scope,
        metadata={},
        installation_id="install",
    )
    assert explicit is not None
    assert explicit["source_seq"] == 9


def _module(name: str) -> ModuleType:
    return _load_module(name, ROOT / "integrations/open-webui/atagia_memory_filter.py")


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
