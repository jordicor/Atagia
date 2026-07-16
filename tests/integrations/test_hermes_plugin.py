from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import inspect
from pathlib import Path
import subprocess
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from atagia.models.schemas_api import ReplaceSelectedTranscriptRequest


ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = ROOT / "integrations/hermes/plugins/memory/atagia"
PATCH_DIR = ROOT / "integrations/hermes/patches/0.18.2-e4ea0a0"
CONTRACT_ROOT = ROOT / "tests/integrations/fixtures/hermes_v0_18_2"
CAPABILITY = "hermes.memory-selection.v1"
HERMES_VERSION = "0.18.2"
HERMES_COMMIT = "e4ea0a0ed7fc24761b2b425146893561a73216e1"


@contextmanager
def _pinned_host() -> tuple[ModuleType, type, type]:
    sys.path.insert(0, str(CONTRACT_ROOT))
    _remove_modules("agent")
    loader = _load_module(
        "hermes_v0182_strict_loader",
        CONTRACT_ROOT / "strict_loader.py",
    )
    patched = _load_module(
        "hermes_v0182_patched_memory_selection_host",
        CONTRACT_ROOT / "patched_memory_selection_host.py",
    )
    from agent.memory_provider import MemoryProvider

    try:
        yield loader, MemoryProvider, patched.PatchedMemorySelectionHost
    finally:
        _remove_modules("_hermes_contract_memory_atagia")
        _remove_modules("hermes_v0182_strict_loader")
        _remove_modules("hermes_v0182_patched_memory_selection_host")
        _remove_modules("agent")
        sys.path.remove(str(CONTRACT_ROOT))


def _configure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATAGIA_BASE_URL", "http://atagia.test")
    monkeypatch.setenv("ATAGIA_SERVICE_API_KEY", "service-key")
    monkeypatch.setenv("ATAGIA_HERMES_INSTALLATION_ID", "hermes-install-01")
    monkeypatch.setenv("ATAGIA_HERMES_HOST_ACCOUNT_ID", "hermes-account-01")
    monkeypatch.setenv("ATAGIA_HERMES_USER_ID", "atagia-user-01")


def _api(
    calls: list[dict],
    *,
    selection_status: str = "complete",
    context_ack: bool = True,
):
    def fake_request_json(path, payload, extra_headers=None):
        calls.append(
            {
                "method": "POST",
                "path": path,
                "payload": dict(payload),
                "headers": dict(extra_headers or {}),
            }
        )
        if path.endswith("/selected-transcript"):
            return {
                "operation_id": payload["operation_id"],
                "selection_epoch": payload["selection_epoch"],
                "status": selection_status,
            }
        if path.endswith("/context"):
            response = {"system_prompt": "Remember the selected context."}
            if context_ack:
                response["request_message_id"] = payload["message_id"]
            return response
        return {
            "message_id": payload["message_id"],
            "source_seq": payload["source_seq"],
        }

    return fake_request_json


def _wait(provider) -> None:
    sys.modules[provider.__class__.__module__].wait_for_queue(provider)


def test_pinned_loader_abi_capability_and_vanilla_rejection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure(monkeypatch)
    with _pinned_host() as (loader, memory_provider_abc, patched_host):
        vanilla = loader.load_registered_provider(PLUGIN_DIR)
        module = sys.modules[vanilla.__class__.__module__]
        assert isinstance(vanilla, memory_provider_abc)
        assert vanilla.name == "atagia"
        assert vanilla.__class__.__abstractmethods__ == frozenset()
        assert vanilla.is_available() is True
        assert list(inspect.signature(vanilla.initialize).parameters) == [
            "session_id",
            "kwargs",
        ]
        with pytest.raises(
            module.UnsupportedHermesHostError,
            match="vanilla host is not mutation-safe",
        ):
            vanilla.initialize(
                "vanilla-session",
                hermes_home=str(tmp_path),
                platform="cli",
                agent_context="primary",
            )
        assert vanilla.is_available() is False
        assert (
            vanilla.status()["error_code"]
            == "hermes_memory_selection_capability_missing"
        )
        loader.unload_provider(vanilla)

        provider = loader.load_registered_provider(PLUGIN_DIR)
        patched_host(
            provider,
            session_id="patched-session",
            hermes_home=tmp_path,
        )
        assert provider.is_available() is True
        assert provider.status()["status"] == "ready"
        loader.unload_provider(provider)

    manifest = (PLUGIN_DIR / "plugin.yaml").read_text(encoding="utf-8")
    assert "name: atagia" in manifest
    assert "version: 0.4.0" in manifest


def test_straight_line_repeated_text_restart_and_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure(monkeypatch)
    calls: list[dict] = []
    with _pinned_host() as (loader, _, patched_host):
        provider = loader.load_registered_provider(PLUGIN_DIR)
        host = patched_host(
            provider,
            session_id="session/with slash",
            hermes_home=tmp_path,
        )
        monkeypatch.setattr(provider, "_request_json", _api(calls))

        assert host.start_turn("yes") == "Remember the selected context."
        host.complete_turn("same answer")
        _wait(provider)
        assert host.start_turn("yes") == "Remember the selected context."
        selected = host.complete_turn("same answer")
        _wait(provider)

        contexts = [call for call in calls if call["path"].endswith("/context")]
        responses = [call for call in calls if call["path"].endswith("/responses")]
        assert [call["payload"]["source_seq"] for call in contexts] == [1, 3]
        assert [call["payload"]["source_seq"] for call in responses] == [2, 4]
        assert contexts[0]["payload"]["message_id"] != contexts[1]["payload"]["message_id"]
        assert responses[0]["payload"]["message_id"] != responses[1]["payload"]["message_id"]

        host.end_session()
        _wait(provider)
        assert provider.status()["status"] == "session_reconciled"
        assert provider.status()["imported"] == 0
        loader.unload_provider(provider)

        restarted = loader.load_registered_provider(PLUGIN_DIR)
        restarted_host = patched_host(
            restarted,
            session_id="session/with slash",
            hermes_home=tmp_path,
            selected_messages=selected,
            next_row_id=5,
            next_turn_number=3,
        )
        monkeypatch.setattr(restarted, "_request_json", _api(calls))
        restarted_host.start_turn("yes")
        restarted_host.complete_turn("same answer")
        _wait(restarted)
        assert calls[-1]["payload"]["source_seq"] == 6
        loader.unload_provider(restarted)


@pytest.mark.parametrize("mutation_kind", ["retry", "regeneration"])
def test_retry_and_regeneration_reconcile_before_prefetch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation_kind: str,
) -> None:
    _configure(monkeypatch)
    calls: list[dict] = []
    with _pinned_host() as (loader, _, patched_host):
        provider = loader.load_registered_provider(PLUGIN_DIR)
        host = patched_host(provider, session_id="retry-session", hermes_home=tmp_path)
        monkeypatch.setattr(provider, "_request_json", _api(calls))

        host.start_turn("q1")
        first_turn = host.complete_turn("a1")
        _wait(provider)
        host.start_turn("q2")
        old_selection = host.complete_turn("old a2")
        _wait(provider)
        calls.clear()

        host.retry_last("q2", mutation_kind=mutation_kind)
        assert calls[0]["path"].endswith("/selected-transcript")
        assert calls[1]["path"].endswith("/context")
        replacement = calls[0]["payload"]
        ReplaceSelectedTranscriptRequest.model_validate(replacement)
        assert replacement["contract_version"] == "atagia.selected-transcript.v1"
        assert replacement["mutation_kind"] == mutation_kind
        assert replacement["selection_epoch"] == 1
        assert [message["text"] for message in replacement["messages"]] == ["q1", "a1"]
        assert replacement["retained_cutoff_message_id"] == calls[0]["payload"]["messages"][-1]["message_id"]
        assert old_selection[-2]["host_message_id"] not in {
            message["host_message_id"] for message in replacement["messages"]
        }

        selected = host.complete_turn("new a2")
        _wait(provider)
        assert [call["payload"]["source_seq"] for call in calls if call["path"].endswith("/context")] == [5]
        assert [call["payload"]["source_seq"] for call in calls if call["path"].endswith("/responses")] == [6]
        assert selected[:2] == first_turn
        assert provider.status()["error_code"] is None
        loader.unload_provider(provider)


def test_undo_reconciles_immediately_and_next_turn_uses_monotonic_sequence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure(monkeypatch)
    calls: list[dict] = []
    with _pinned_host() as (loader, _, patched_host):
        provider = loader.load_registered_provider(PLUGIN_DIR)
        host = patched_host(provider, session_id="undo-session", hermes_home=tmp_path)
        monkeypatch.setattr(provider, "_request_json", _api(calls))
        for index in range(1, 3):
            host.start_turn(f"q{index}")
            host.complete_turn(f"a{index}")
            _wait(provider)
        calls.clear()

        host.undo()
        assert len(calls) == 1
        assert calls[0]["path"].endswith("/selected-transcript")
        assert calls[0]["payload"]["mutation_kind"] == "undo"
        assert [message["text"] for message in calls[0]["payload"]["messages"]] == ["q1", "a1"]
        assert provider.is_available() is True

        host.start_turn("replacement q2")
        host.complete_turn("replacement a2")
        _wait(provider)
        context = next(call for call in calls if call["path"].endswith("/context"))
        response = next(call for call in calls if call["path"].endswith("/responses"))
        assert (context["payload"]["source_seq"], response["payload"]["source_seq"]) == (5, 6)
        loader.unload_provider(provider)


def test_rebuilding_selection_is_polled_before_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure(monkeypatch)
    calls: list[dict] = []
    with _pinned_host() as (loader, _, patched_host):
        provider = loader.load_registered_provider(PLUGIN_DIR)
        host = patched_host(provider, session_id="poll-session", hermes_home=tmp_path)
        monkeypatch.setattr(provider, "_request_json", _api(calls))
        host.start_turn("q1")
        host.complete_turn("a1")
        _wait(provider)
        host.start_turn("q2")
        host.complete_turn("a2")
        _wait(provider)
        calls.clear()
        monkeypatch.setattr(
            provider,
            "_request_json",
            _api(calls, selection_status="rebuilding"),
        )

        def fake_get(path, *, conversation_id):
            calls.append({"method": "GET", "path": path, "conversation_id": conversation_id})
            snapshot = provider._store.selection_snapshot(provider._identity_scope, host.session_id)
            assert snapshot is not None
            return {
                "operation_id": snapshot.operation_id,
                "selection_epoch": snapshot.selection_epoch,
                "status": "complete",
            }

        monkeypatch.setattr(provider, "_get_json", fake_get)
        host.retry_last("q2")
        assert [call["method"] for call in calls[:3]] == ["POST", "GET", "POST"]
        assert calls[0]["path"].endswith("/selected-transcript")
        assert calls[2]["path"].endswith("/context")
        loader.unload_provider(provider)


def test_remediation_required_selection_blocks_context_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure(monkeypatch)
    calls: list[dict] = []
    with _pinned_host() as (loader, _, patched_host):
        provider = loader.load_registered_provider(PLUGIN_DIR)
        host = patched_host(provider, session_id="remediation-session", hermes_home=tmp_path)
        monkeypatch.setattr(provider, "_request_json", _api(calls))
        host.start_turn("q1")
        host.complete_turn("a1")
        _wait(provider)
        host.start_turn("q2")
        host.complete_turn("a2")
        _wait(provider)
        calls.clear()
        monkeypatch.setattr(
            provider,
            "_request_json",
            _api(calls, selection_status="remediation_required"),
        )

        assert host.retry_last("q2") == ""
        assert len(calls) == 1
        assert calls[0]["path"].endswith("/selected-transcript")
        assert provider.is_available() is False
        assert provider.status()["error_code"] == "hermes_selected_transcript_remediation_required"
        loader.unload_provider(provider)


def test_missing_or_invalid_turn_signal_prevents_any_remote_effect(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure(monkeypatch)
    calls: list[dict] = []
    with _pinned_host() as (loader, _, patched_host):
        provider = loader.load_registered_provider(PLUGIN_DIR)
        patched_host(provider, session_id="signal-session", hermes_home=tmp_path)
        monkeypatch.setattr(provider, "_request_json", _api(calls))

        provider.on_turn_start(1, "q1", session_id="signal-session")
        assert provider.prefetch("q1", session_id="signal-session") == ""
        assert calls == []
        assert provider.is_available() is False
        assert provider.status()["error_code"] == "hermes_memory_selection_signal_invalid"
        loader.unload_provider(provider)


def test_empty_assistant_boundary_is_not_selected_or_reconciled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure(monkeypatch)
    calls: list[dict] = []
    with _pinned_host() as (loader, _, patched_host):
        provider = loader.load_registered_provider(PLUGIN_DIR)
        host = patched_host(provider, session_id="empty-assistant", hermes_home=tmp_path)
        monkeypatch.setattr(provider, "_request_json", _api(calls))

        host.start_turn("q1")
        host.complete_turn("")

        assert provider.is_available() is False
        assert provider.status()["error_code"] == (
            "hermes_selected_transcript_unverifiable"
        )
        assert not any(call["path"].endswith("/responses") for call in calls)
        before = list(calls)
        host.end_session()
        assert calls == before
        assert host.start_turn("q2") == ""
        assert calls == before
        loader.unload_provider(provider)


@pytest.mark.parametrize("mutation_kind", ["retry", "undo"])
def test_orphan_user_after_rewind_is_rejected_before_mutation_or_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation_kind: str,
) -> None:
    _configure(monkeypatch)
    calls: list[dict] = []
    with _pinned_host() as (loader, _, patched_host):
        provider = loader.load_registered_provider(PLUGIN_DIR)
        host = patched_host(provider, session_id="orphan-user", hermes_home=tmp_path)
        monkeypatch.setattr(provider, "_request_json", _api(calls))
        host.start_turn("q1")
        host.complete_turn("a1")
        _wait(provider)
        host.start_turn("q2")
        host.complete_turn("a2")
        _wait(provider)
        calls.clear()

        retained = host.selected_messages[:2]
        orphan = {
            "role": "user",
            "content": "q2",
            "host_message_id": "hermes-sqlite-message:5",
            "generation_id": "default",
        }
        signal = {
            "contract_version": CAPABILITY,
            "session_id": host.session_id,
            "mutation_kind": mutation_kind,
            "retained_cutoff_host_message_id": orphan["host_message_id"],
            "messages": [*retained, orphan],
            "current_user_message": None,
        }
        if mutation_kind == "retry":
            current = {
                "role": "user",
                "content": "q2",
                "host_message_id": "hermes-sqlite-message:6",
                "generation_id": "default",
            }
            signal["current_user_message"] = current
            provider.on_turn_start(
                3,
                "q2",
                session_id=host.session_id,
                memory_selection=signal,
            )
            assert provider.prefetch("q2", session_id=host.session_id) == ""
        else:
            provider.on_session_switch(
                host.session_id,
                rewound=True,
                memory_selection=signal,
            )

        assert calls == []
        assert provider.is_available() is False
        assert provider.status()["error_code"] == (
            "hermes_memory_selection_signal_invalid"
        )
        loader.unload_provider(provider)


def test_first_attach_to_advanced_session_is_rejected_before_prefetch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure(monkeypatch)
    calls: list[dict] = []
    history = [
        {"role": "user", "content": "q1", "host_message_id": "hermes-sqlite-message:1", "generation_id": "default"},
        {"role": "assistant", "content": "a1", "host_message_id": "hermes-sqlite-message:2", "generation_id": "default"},
    ]
    with _pinned_host() as (loader, _, patched_host):
        provider = loader.load_registered_provider(PLUGIN_DIR)
        host = patched_host(
            provider,
            session_id="advanced-session",
            hermes_home=tmp_path,
            selected_messages=history,
            next_row_id=3,
            next_turn_number=2,
        )
        monkeypatch.setattr(provider, "_request_json", _api(calls))
        assert host.start_turn("q2") == ""
        assert calls == []
        assert provider.status()["error_code"] == "hermes_mid_session_attach_unsupported"
        loader.unload_provider(provider)


def test_context_requires_exact_request_identity_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure(monkeypatch)
    calls: list[dict] = []
    with _pinned_host() as (loader, _, patched_host):
        provider = loader.load_registered_provider(PLUGIN_DIR)
        host = patched_host(provider, session_id="ack-session", hermes_home=tmp_path)
        monkeypatch.setattr(provider, "_request_json", _api(calls, context_ack=False))
        assert host.start_turn("q1") == ""
        assert provider.status()["status"] == "failed_open"
        host.complete_turn("a1")
        _wait(provider)
        assert [call["path"] for call in calls].count("/v1/conversations/ack-session/context") == 1
        loader.unload_provider(provider)


def test_tool_protocol_uses_only_final_assistant_identity_and_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure(monkeypatch)
    calls: list[dict] = []
    with _pinned_host() as (loader, _, patched_host):
        provider = loader.load_registered_provider(PLUGIN_DIR)
        host = patched_host(provider, session_id="tool-session", hermes_home=tmp_path)
        monkeypatch.setattr(provider, "_request_json", _api(calls))
        host.start_turn("inspect")
        assert host.current_user is not None
        messages = [
            host.current_user,
            {
                "role": "assistant",
                "content": "calling tool",
                "host_message_id": "hermes-sqlite-message:2",
                "generation_id": "default",
                "tool_calls": [{"id": "call-1"}],
            },
            {"role": "tool", "content": "result"},
            {
                "role": "assistant",
                "content": "final answer",
                "host_message_id": "hermes-sqlite-message:3",
                "generation_id": "default",
            },
        ]
        host.complete_turn("final answer", row_id=3, messages=messages)
        _wait(provider)
        response = next(call for call in calls if call["path"].endswith("/responses"))
        assert response["payload"]["text"] == "final answer"
        assert response["payload"]["source_seq"] == 2
        loader.unload_provider(provider)


def test_multimodal_turn_without_prefetch_uses_authoritative_host_projection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure(monkeypatch)
    calls: list[dict] = []
    with _pinned_host() as (loader, _, patched_host):
        provider = loader.load_registered_provider(PLUGIN_DIR)
        host = patched_host(provider, session_id="multimodal-session", hermes_home=tmp_path)
        monkeypatch.setattr(provider, "_request_json", _api(calls))
        host.start_turn("[screenshot] what is shown?", prefetch=False)
        host.complete_turn("a terminal window")
        _wait(provider)
        assert [call["payload"]["text"] for call in calls] == [
            "[screenshot] what is shown?",
            "a terminal window",
        ]
        assert [call["payload"]["source_seq"] for call in calls] == [1, 2]
        loader.unload_provider(provider)


def test_assistant_delivery_failure_is_retried_by_authoritative_session_end(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure(monkeypatch)
    calls: list[dict] = []
    failed = False

    def flaky(path, payload, extra_headers=None):
        nonlocal failed
        calls.append({"path": path, "payload": dict(payload)})
        if path.endswith("/context"):
            return {
                "system_prompt": "context",
                "request_message_id": payload["message_id"],
            }
        if path.endswith("/responses") and not failed:
            failed = True
            raise RuntimeError("temporary failure")
        return {"message_id": payload["message_id"], "source_seq": payload["source_seq"]}

    with _pinned_host() as (loader, _, patched_host):
        provider = loader.load_registered_provider(PLUGIN_DIR)
        host = patched_host(provider, session_id="retry-delivery", hermes_home=tmp_path)
        monkeypatch.setattr(provider, "_request_json", flaky)
        host.start_turn("q1")
        host.complete_turn("a1")
        _wait(provider)
        assert provider.status()["status"] == "worker_failed_open"
        host.end_session()
        _wait(provider)
        attempts = [call for call in calls if call["path"].endswith("/responses")]
        assert len(attempts) == 2
        assert (
            attempts[0]["payload"]["message_id"],
            attempts[0]["payload"]["source_seq"],
        ) == (
            attempts[1]["payload"]["message_id"],
            attempts[1]["payload"]["source_seq"],
        )
        assert provider.status()["status"] == "session_reconciled"
        loader.unload_provider(provider)


def test_bounded_shutdown_keeps_store_owned_until_worker_drains(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure(monkeypatch)
    calls: list[dict] = []
    entered = threading.Event()
    release = threading.Event()
    with _pinned_host() as (loader, _, patched_host):
        provider = loader.load_registered_provider(PLUGIN_DIR)
        provider.config.timeout_seconds = 0.01
        host = patched_host(provider, session_id="shutdown-session", hermes_home=tmp_path)
        monkeypatch.setattr(provider, "_request_json", _api(calls))
        original = provider._sync_turn_now

        def blocking(item):
            entered.set()
            assert release.wait(timeout=5)
            original(item)

        monkeypatch.setattr(provider, "_sync_turn_now", blocking)
        host.start_turn("q1")
        host.complete_turn("a1")
        assert entered.wait(timeout=1)
        provider.shutdown()
        assert provider.status()["status"] == "shutdown_draining"
        assert provider._store is not None
        release.set()
        _wait(provider)
        assert provider.status()["status"] == "shutdown"
        assert provider._store is None
        loader.unload_provider(provider)


def test_canonical_identity_is_globally_namespaced_and_retry_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure(monkeypatch)
    with _pinned_host() as (loader, _, _):
        provider = loader.load_registered_provider(PLUGIN_DIR)
        module = sys.modules[provider.__class__.__module__]
        base = {
            "integration_kind": "hermes",
            "host_installation_id": "install",
            "host_account_id": "account",
            "user_id": "user",
            "host_conversation_id": "session",
            "source_namespace": "host_message",
            "host_message_id": "sqlite-row-1",
            "role": "assistant",
            "generation_id": "default",
        }
        identity = module.canonical_external_message_id(**base)
        assert identity == module.canonical_external_message_id(**base)
        for field, value in (
            ("integration_kind", "openclaw"),
            ("host_installation_id", "install-2"),
            ("host_account_id", "account-2"),
            ("user_id", "user-2"),
            ("host_conversation_id", "session-2"),
            ("host_message_id", "sqlite-row-2"),
            ("role", "user"),
            ("generation_id", "generation-2"),
        ):
            changed = dict(base)
            changed[field] = value
            assert module.canonical_external_message_id(**changed) != identity
        loader.unload_provider(provider)


def test_downstream_patch_is_pinned_and_embeds_the_module_source() -> None:
    patch_path = PATCH_DIR / "hermes-memory-selection-v1.patch"
    metadata_path = PATCH_DIR / "PATCH_METADATA.json"
    assert patch_path.is_file()
    assert metadata_path.is_file()
    patch = patch_path.read_text(encoding="utf-8")
    assert CAPABILITY in patch
    assert HERMES_COMMIT in patch
    for target in (
        "agent/agent_init.py",
        "agent/turn_context.py",
        "agent/memory_selection.py",
        "run_agent.py",
        "cli.py",
    ):
        assert f"b/{target}" in patch

    module_diff = patch.split(
        "diff --git a/agent/memory_selection.py b/agent/memory_selection.py\n",
        1,
    )[1].split("\ndiff --git ", 1)[0]
    embedded_lines = [
        line[1:]
        for line in module_diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    assert "\n".join(embedded_lines) + "\n" == (
        PATCH_DIR / "memory_selection.py"
    ).read_text(encoding="utf-8")


def test_downstream_patch_reverses_against_exact_clean_pinned_checkout() -> None:
    checkout = Path("/tmp/hermes-agent-review")
    if not checkout.is_dir():
        pytest.skip(f"pinned Hermes checkout is not available at {checkout}")
    patch_path = PATCH_DIR / "hermes-memory-selection-v1.patch"
    assert subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip() == HERMES_COMMIT
    reverse_check = subprocess.run(
        ["git", "-C", str(checkout), "apply", "--check", "--reverse", str(patch_path)],
        capture_output=True,
        text=True,
    )
    assert reverse_check.returncode == 0, reverse_check.stderr


def test_patched_host_module_emits_durable_selection_before_prefetch() -> None:
    module = _load_module(
        "hermes_memory_selection_patch_source",
        PATCH_DIR / "memory_selection.py",
    )

    class Database:
        def get_messages(self, session_id, include_inactive=False):
            assert session_id == "host-session"
            assert include_inactive is False
            return [
                {"id": 1, "role": "user", "content": "q1"},
                {"id": 2, "role": "assistant", "content": "a1"},
                {"id": 3, "role": "user", "content": "q2"},
            ]

    agent = SimpleNamespace(session_id="host-session", _session_db=Database())
    assert module.host_capabilities()[CAPABILITY]["hermes_commit"] == HERMES_COMMIT
    signal = module.build_turn_signal(agent)
    assert signal["contract_version"] == CAPABILITY
    assert signal["retained_cutoff_host_message_id"] == "hermes-sqlite-message:2"
    assert [row["host_message_id"] for row in signal["messages"]] == [
        "hermes-sqlite-message:1",
        "hermes-sqlite-message:2",
    ]
    assert signal["current_user_message"]["host_message_id"] == (
        "hermes-sqlite-message:3"
    )
    module.mark_next_turn_mutation(agent, "regeneration")
    assert module.build_turn_signal(agent)["mutation_kind"] == "regeneration"
    _remove_modules("hermes_memory_selection_patch_source")


def test_patch_source_surfaces_empty_and_orphan_boundaries_with_row_ids() -> None:
    module = _load_module(
        "hermes_memory_selection_boundary_source",
        PATCH_DIR / "memory_selection.py",
    )

    class Database:
        rows: list[dict] = []

        def get_messages(self, session_id, include_inactive=False):
            assert session_id == "boundary-session"
            assert include_inactive is False
            return list(self.rows)

    database = Database()
    agent = SimpleNamespace(
        session_id="boundary-session",
        _session_db=database,
    )
    database.rows = [
        {"id": 41, "role": "user", "content": " q1 "},
        {"id": 42, "role": "assistant", "content": ""},
    ]
    boundary = module.selected_transcript(agent)
    assert [row["host_message_id"] for row in boundary] == [
        "hermes-sqlite-message:41",
        "hermes-sqlite-message:42",
    ]
    assert [row["content"] for row in boundary] == [" q1 ", ""]
    with pytest.raises(RuntimeError, match="message text is unavailable"):
        module.build_rewind_signal(agent, "undo")

    database.rows = [{"id": 43, "role": "user", "content": "orphan"}]
    assert module.selected_transcript(agent)[0]["host_message_id"] == (
        "hermes-sqlite-message:43"
    )
    with pytest.raises(RuntimeError, match="complete turns"):
        module.build_rewind_signal(agent, "retry")

    database.rows = [
        {"id": 44, "role": "user", "content": "use tool"},
        {
            "id": 45,
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1"}],
        },
        {"id": 46, "role": "tool", "content": "result"},
        {"id": 47, "role": "assistant", "content": "done"},
    ]
    selected = module.selected_transcript(agent)
    assert [row["host_message_id"] for row in selected] == [
        "hermes-sqlite-message:44",
        "hermes-sqlite-message:47",
    ]
    _remove_modules("hermes_memory_selection_boundary_source")


def test_provider_does_not_import_without_real_or_faithful_hermes_host() -> None:
    _remove_modules("agent")
    with pytest.raises(RuntimeError, match="requires Hermes Agent 0.18.2"):
        _load_module("atagia_hermes_without_host", PLUGIN_DIR / "provider.py")


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _remove_modules(prefix: str) -> None:
    for name in list(sys.modules):
        if name == prefix or name.startswith(f"{prefix}."):
            sys.modules.pop(name, None)
