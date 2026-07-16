from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
from typing import Iterator

from fastapi.testclient import TestClient
import pytest

from atagia.app import create_app
from atagia.core.config import Settings
from integrations.importers.atagia_importers import canonical_external_message_id


ROOT = Path(__file__).resolve().parents[2]
OPENCLAW_ROOT = ROOT.parent / "openclaw"
PLUGIN_ROOT = ROOT / "integrations" / "openclaw" / "plugin"
LOADER_SMOKE = (
    ROOT / "tests" / "integrations" / "fixtures" / "openclaw_2026_5_6_loader_smoke.mjs"
)
SUPPORTED_VERSION = "2026.5.6"
SUPPORTED_COMMIT = "8934095c828de8d6268e0e42d8cfe6651ccf5a1b"
MIGRATIONS_DIR = ROOT / "src" / "atagia" / "resources" / "migrations"
MANIFESTS_DIR = ROOT / "src" / "atagia" / "resources" / "manifests"


class _TestClientProxyHandler(BaseHTTPRequestHandler):
    server: "_TestClientProxyServer"

    def _forward(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else None
        response = self.server.client.request(
            self.command,
            self.path,
            headers=dict(self.headers.items()),
            content=body,
        )
        self.send_response(response.status_code)
        self.send_header(
            "Content-Type", response.headers.get("content-type", "application/json")
        )
        self.send_header("Content-Length", str(len(response.content)))
        if retry_after := response.headers.get("retry-after"):
            self.send_header("Retry-After", retry_after)
        self.end_headers()
        self.wfile.write(response.content)

    do_GET = _forward
    do_POST = _forward

    def log_message(self, format: str, *args: object) -> None:
        del format, args


class _TestClientProxyServer(ThreadingHTTPServer):
    def __init__(self, client: TestClient) -> None:
        super().__init__(("127.0.0.1", 0), _TestClientProxyHandler)
        self.client = client


@contextmanager
def _serve_test_client(client: TestClient) -> Iterator[str]:
    server = _TestClientProxyServer(client)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _api_settings(tmp_path: Path) -> Settings:
    return Settings(
        sqlite_path=str(tmp_path / "openclaw-real-api.db"),
        migrations_path=str(MIGRATIONS_DIR),
        manifests_path=str(MANIFESTS_DIR),
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        openai_api_key="test-openai-key",
        openrouter_api_key=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
        llm_chat_model="openai/test-model",
        llm_forced_global_model="openai/test-model",
        service_mode=True,
        service_api_key="service-secret",
        admin_api_key="admin-secret",
        workers_enabled=False,
        lifecycle_worker_enabled=False,
        debug=False,
    )


def _require_pinned_openclaw() -> None:
    if not (OPENCLAW_ROOT / "openclaw.mjs").is_file():
        pytest.skip("local OpenClaw checkout is unavailable")
    package = json.loads((OPENCLAW_ROOT / "package.json").read_text(encoding="utf-8"))
    if package.get("version") != SUPPORTED_VERSION:
        pytest.fail(
            f"OpenClaw {SUPPORTED_VERSION} is required; found {package.get('version')!r}"
        )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=OPENCLAW_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != SUPPORTED_COMMIT:
        pytest.fail(f"OpenClaw commit {SUPPORTED_COMMIT} is required; found {commit}")


def _plugin_config() -> dict[str, object]:
    return {
        "plugins": {
            "allow": ["atagia-memory"],
            "load": {"paths": [str(PLUGIN_ROOT)]},
            "entries": {
                "atagia-memory": {
                    "enabled": True,
                    "hooks": {"allowConversationAccess": True},
                    "config": {
                        "baseUrl": "http://127.0.0.1:8100",
                        "apiKey": "contract-only-key",
                        "installationId": "contract-installation",
                        "hostAccountId": "contract-account",
                        "userId": "contract-atagia-user",
                    },
                }
            },
        }
    }


def test_openclaw_manifest_and_package_declare_the_pinned_host() -> None:
    _require_pinned_openclaw()
    manifest = json.loads(
        (PLUGIN_ROOT / "openclaw.plugin.json").read_text(encoding="utf-8")
    )
    package = json.loads((PLUGIN_ROOT / "package.json").read_text(encoding="utf-8"))

    assert manifest["id"] == "atagia-memory"
    assert manifest["activation"] == {"onStartup": True}
    assert manifest["configSchema"]["additionalProperties"] is False
    assert manifest["configSchema"]["properties"]["selectedTranscriptWaitMs"] == {
        "type": "integer",
        "minimum": 100,
        "maximum": 25000,
    }
    assert manifest["configSchema"]["properties"][
        "selectedTranscriptPollIntervalMs"
    ] == {
        "type": "integer",
        "minimum": 25,
        "maximum": 2000,
    }
    assert package["openclaw"]["extensions"] == ["./index.js"]
    assert package["openclaw"]["runtimeExtensions"] == ["./index.js"]
    assert package["openclaw"]["install"]["minHostVersion"] == f">={SUPPORTED_VERSION}"
    assert package["openclaw"]["compat"]["pluginApi"] == SUPPORTED_VERSION
    assert package["openclaw"]["build"]["openclawVersion"] == SUPPORTED_VERSION


def test_official_openclaw_loader_runs_turn_shutdown_and_reload() -> None:
    _require_pinned_openclaw()
    completed = subprocess.run(
        [
            "node",
            str(LOADER_SMOKE),
            str(OPENCLAW_ROOT),
            str(PLUGIN_ROOT),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result == {
        "status": "loaded",
        "hookNames": [
            "before_prompt_build",
            "agent_end",
            "before_compaction",
            "before_reset",
            "session_end",
        ],
        "lifecycleRequests": 4,
        "reloadStable": True,
    }


def test_openclaw_runtime_inspect_reports_registered_hooks(tmp_path: Path) -> None:
    _require_pinned_openclaw()
    state_dir = tmp_path / "openclaw-state"
    state_dir.mkdir()
    (state_dir / "openclaw.json").write_text(
        json.dumps(_plugin_config()),
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "OPENCLAW_STATE_DIR": str(state_dir),
        "OPENCLAW_DISABLE_BUNDLED_PLUGINS": "1",
        "NO_COLOR": "1",
    }
    completed = subprocess.run(
        [
            "node",
            str(OPENCLAW_ROOT / "openclaw.mjs"),
            "plugins",
            "inspect",
            "atagia-memory",
            "--runtime",
            "--json",
        ],
        cwd=OPENCLAW_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    report = json.loads(completed.stdout)
    assert report["plugin"]["id"] == "atagia-memory"
    assert report["plugin"]["status"] == "loaded"
    assert {entry["name"] for entry in report["typedHooks"]} == {
        "before_prompt_build",
        "agent_end",
        "before_compaction",
        "before_reset",
        "session_end",
    }
    assert report["policy"]["allowConversationAccess"] is True


def test_openclaw_runtime_and_offline_importer_share_canonical_identity() -> None:
    identity = {
        "integrationKind": "openclaw",
        "installationId": "installation-日本語",
        "hostAccountId": "account-1",
        "mappedUserId": "user-1",
        "hostConversationId": "agent:main:chat/1",
        "sourceNamespace": "host_message",
        "hostMessageId": "entry-7",
        "role": "assistant",
        "generationId": "generation-2",
    }
    script = """
import { canonicalExternalMessageId } from './integrations/openclaw/plugin/index.js';
const identity = JSON.parse(process.argv[1]);
process.stdout.write(canonicalExternalMessageId(identity));
"""
    completed = subprocess.run(
        ["node", "--input-type=module", "--eval", script, json.dumps(identity)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.stdout == canonical_external_message_id(
        integration_kind=identity["integrationKind"],
        host_installation_id=identity["installationId"],
        host_account_id=identity["hostAccountId"],
        user_id=identity["mappedUserId"],
        host_conversation_id=identity["hostConversationId"],
        source_namespace=identity["sourceNamespace"],
        host_message_id=identity["hostMessageId"],
        role=identity["role"],
        generation_id=identity["generationId"],
    )


def test_openclaw_selected_branch_reaches_real_atagia_api(tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    app = create_app(_api_settings(tmp_path))
    service_headers = {
        "Authorization": "Bearer service-secret",
        "X-Atagia-User-Id": "atagia-user",
        "X-Atagia-Platform-Id": "openclaw",
    }
    conversation_id = "agent:main:real-api"
    with TestClient(app) as client:
        assert (
            client.post(
                "/v1/users",
                headers=service_headers,
                json={"user_id": "atagia-user"},
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/v1/conversations",
                headers=service_headers,
                json={
                    "user_id": "atagia-user",
                    "conversation_id": conversation_id,
                    "platform_id": "openclaw",
                    "mode": "general_qa",
                },
            ).status_code
            == 200
        )
        runtime = client.app.state.runtime
        runtime.settings = replace(runtime.settings, workers_enabled=True)
        state_dir = tmp_path / "openclaw-state"
        script = """
import { AtagiaOpenClawRuntime } from './integrations/openclaw/plugin/index.js';
import fs from 'node:fs';
import path from 'node:path';
const [baseUrl, stateDir, conversationId, assistantId, assistantText] = process.argv.slice(1);
const runtime = new AtagiaOpenClawRuntime({
  pluginConfig: {
    baseUrl,
    apiKey: 'service-secret',
    installationId: 'real-api-installation',
    hostAccountId: 'real-api-account',
    userId: 'atagia-user',
    mode: 'general_qa',
    failOpen: false,
    selectedTranscriptWaitMs: 100,
    selectedTranscriptPollIntervalMs: 25,
  },
  env: {},
  logger: { info() {}, warn() {} },
});
runtime.start({ stateDir });
const sessionFile = path.join(stateDir, 'branch-navigation.jsonl');
fs.writeFileSync(sessionFile, `${[
  { type: 'session', id: conversationId, version: 3 },
  { type: 'message', id: 'real-user-message', parentId: null,
    message: { role: 'user', content: 'Selected question' } },
  { type: 'message', id: 'real-assistant-v1', parentId: 'real-user-message',
    message: { role: 'assistant', content: 'Selected answer one' } },
  { type: 'message', id: 'real-assistant-v2', parentId: 'real-user-message',
    message: { role: 'assistant', content: 'Selected answer two' } },
].map((entry) => JSON.stringify(entry)).join('\\n')}\\n`);
await runtime.backfill({ sessionFile, messages: [
  { role: 'user', content: 'Selected question' },
  { id: assistantId, role: 'assistant', content: assistantText },
] }, { sessionKey: 'agent:main:real-api-route', sessionId: conversationId }, 'before_compaction');
const status = runtime.getStatus();
runtime.stop();
process.stdout.write(JSON.stringify(status));
"""
        with _serve_test_client(client) as base_url:

            def run_adapter(
                assistant_id: str,
                assistant_text: str,
            ) -> dict[str, object]:
                completed = subprocess.run(
                    [
                        node,
                        "--input-type=module",
                        "--eval",
                        script,
                        base_url,
                        str(state_dir),
                        conversation_id,
                        assistant_id,
                        assistant_text,
                    ],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                return json.loads(completed.stdout)

            first_status = run_adapter("real-assistant-v1", "Selected answer one")
            assert first_status["status"] == "selected_transcript_rebuilding"
            assert first_status["selectionEpoch"] == 0
            assert first_status["selectedMessageCount"] == 2

            connection = client.portal.call(runtime.open_connection)
            try:
                cursor = client.portal.call(
                    connection.execute,
                    """
                    SELECT id, role, seq
                    FROM messages
                    WHERE conversation_id = ?
                    ORDER BY seq ASC
                    """,
                    (conversation_id,),
                )
                first_messages = [
                    tuple(row) for row in client.portal.call(cursor.fetchall)
                ]
                cursor = client.portal.call(
                    connection.execute,
                    """
                    SELECT operation_id, selection_epoch, stage
                    FROM transcript_rebuild_workflows
                    WHERE user_id = ? AND conversation_id = ?
                    """,
                    ("atagia-user", conversation_id),
                )
                first_workflow = tuple(client.portal.call(cursor.fetchone))
                client.portal.call(
                    connection.execute,
                    """
                    UPDATE transcript_rebuild_workflows
                    SET stage = 'complete', completed_at = updated_at
                    WHERE operation_id = ?
                    """,
                    (first_status["operationId"],),
                )
                client.portal.call(
                    connection.execute,
                    """
                    UPDATE conversation_transcript_selections
                    SET state = 'complete'
                    WHERE user_id = ? AND conversation_id = ?
                    """,
                    ("atagia-user", conversation_id),
                )
                client.portal.call(connection.commit)
            finally:
                client.portal.call(connection.close)
            assert [(role, seq) for _, role, seq in first_messages] == [
                ("user", 1),
                ("assistant", 2),
            ]
            assert all(
                message_id.startswith("extmsg_") for message_id, _, _ in first_messages
            )
            assert first_workflow == (
                first_status["operationId"],
                0,
                "preparing",
            )

            second_status = run_adapter("real-assistant-v2", "Selected answer two")
            assert second_status["status"] == "selected_transcript_rebuilding"
            assert second_status["selectionEpoch"] == 1
            assert second_status["operationId"] != first_status["operationId"]

        connection = client.portal.call(runtime.open_connection)
        try:
            cursor = client.portal.call(
                connection.execute,
                """
                SELECT role, text, seq
                FROM messages
                WHERE conversation_id = ?
                ORDER BY seq ASC
                """,
                (conversation_id,),
            )
            selected_messages = [
                tuple(row) for row in client.portal.call(cursor.fetchall)
            ]
            cursor = client.portal.call(
                connection.execute,
                """
                SELECT operation_id, selection_epoch, mutation_kind, stage
                FROM transcript_rebuild_workflows
                WHERE user_id = ? AND conversation_id = ?
                ORDER BY selection_epoch DESC
                LIMIT 1
                """,
                ("atagia-user", conversation_id),
            )
            latest_workflow = tuple(client.portal.call(cursor.fetchone))
        finally:
            client.portal.call(connection.close)
        assert selected_messages == [
            ("user", "Selected question", 1),
            ("assistant", "Selected answer two", 3),
        ]
        assert latest_workflow == (
            second_status["operationId"],
            1,
            "regeneration",
            "preparing",
        )


def test_openclaw_reset_keeps_old_session_in_real_atagia_api(tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    app = create_app(_api_settings(tmp_path))
    headers = {
        "Authorization": "Bearer service-secret",
        "X-Atagia-User-Id": "atagia-user",
        "X-Atagia-Platform-Id": "openclaw",
    }
    session_ids = ("session-before-reset", "session-after-reset")
    with TestClient(app) as client:
        assert (
            client.post(
                "/v1/users",
                headers=headers,
                json={"user_id": "atagia-user"},
            ).status_code
            == 200
        )
        for session_id in session_ids:
            assert (
                client.post(
                    "/v1/conversations",
                    headers=headers,
                    json={
                        "user_id": "atagia-user",
                        "conversation_id": session_id,
                        "platform_id": "openclaw",
                        "mode": "general_qa",
                    },
                ).status_code
                == 200
            )
        runtime = client.app.state.runtime
        runtime.settings = replace(runtime.settings, workers_enabled=True)
        state_dir = tmp_path / "openclaw-reset-state"
        script = """
import { AtagiaOpenClawRuntime } from './integrations/openclaw/plugin/index.js';
const [baseUrl, stateDir, sessionId, answer] = process.argv.slice(1);
const runtime = new AtagiaOpenClawRuntime({
  pluginConfig: {
    baseUrl,
    apiKey: 'service-secret',
    installationId: 'reset-installation',
    hostAccountId: 'reset-account',
    userId: 'atagia-user',
    mode: 'general_qa',
    failOpen: false,
    selectedTranscriptWaitMs: 100,
    selectedTranscriptPollIntervalMs: 25,
  },
  env: {},
  logger: { info() {}, warn() {} },
});
runtime.start({ stateDir });
await runtime.backfill({ sessionId, messages: [
  { id: 'same-host-user', role: 'user', content: 'Question' },
  { id: 'same-host-assistant', role: 'assistant', content: answer },
] }, { sessionKey: 'agent:main:reused-route', sessionId }, 'session_end');
const status = runtime.getStatus();
runtime.stop();
process.stdout.write(JSON.stringify(status));
"""
        with _serve_test_client(client) as base_url:

            def run_session(session_id: str, answer: str) -> dict[str, object]:
                completed = subprocess.run(
                    [
                        node,
                        "--input-type=module",
                        "--eval",
                        script,
                        base_url,
                        str(state_dir),
                        session_id,
                        answer,
                    ],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                return json.loads(completed.stdout)

            old_status = run_session(session_ids[0], "Old session answer")
            connection = client.portal.call(runtime.open_connection)
            try:
                client.portal.call(
                    connection.execute,
                    """
                    UPDATE transcript_rebuild_workflows
                    SET stage = 'complete', completed_at = updated_at
                    WHERE user_id = ? AND conversation_id = ?
                    """,
                    ("atagia-user", session_ids[0]),
                )
                client.portal.call(
                    connection.execute,
                    """
                    UPDATE conversation_transcript_selections
                    SET state = 'complete'
                    WHERE user_id = ? AND conversation_id = ?
                    """,
                    ("atagia-user", session_ids[0]),
                )
                client.portal.call(connection.commit)
            finally:
                client.portal.call(connection.close)
            new_status = run_session(session_ids[1], "New session answer")

        assert old_status["selectionEpoch"] == 0
        assert new_status["selectionEpoch"] == 0
        assert old_status["operationId"] != new_status["operationId"]
        connection = client.portal.call(runtime.open_connection)
        try:
            cursor = client.portal.call(
                connection.execute,
                """
                SELECT m.conversation_id, m.id, m.role, m.text
                FROM messages AS m
                JOIN conversations AS c ON c.id = m.conversation_id
                WHERE c.user_id = ?
                  AND m.conversation_id IN (?, ?)
                ORDER BY m.conversation_id ASC, m.seq ASC
                """,
                ("atagia-user", *session_ids),
            )
            rows = [tuple(row) for row in client.portal.call(cursor.fetchall)]
        finally:
            client.portal.call(connection.close)
        by_session = {
            session_id: [row for row in rows if row[0] == session_id]
            for session_id in session_ids
        }
        assert [(role, text) for _, _, role, text in by_session[session_ids[0]]] == [
            ("user", "Question"),
            ("assistant", "Old session answer"),
        ]
        assert [(role, text) for _, _, role, text in by_session[session_ids[1]]] == [
            ("user", "Question"),
            ("assistant", "New session answer"),
        ]
        assert by_session[session_ids[0]][0][1] != by_session[session_ids[1]][0][1]
