"""Real boundary tests for the SillyTavern server-side credential adapter."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import secrets
import select
import shutil
import socket
import subprocess
import threading
import time
from typing import Iterator

import httpx
import pytest
import uvicorn

from atagia.app import create_app
from atagia.core.config import Settings


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_RESOURCES = ROOT / "src" / "atagia" / "resources"
SERVER_PLUGIN = ROOT / "integrations" / "sillytavern" / "server-plugin" / "index.cjs"
NODE_HARNESS = (
    ROOT / "tests" / "integrations" / "fixtures" / "sillytavern_server_harness.cjs"
)
SILLYTAVERN_CHECKOUT = ROOT.parent / "sillytavern"
SILLYTAVERN_VERSION = "1.18.0"
SILLYTAVERN_COMMIT = "51ad27fb86d39a3daca3adaa970375c9670c12df"


def _credential(label: str) -> str:
    return f"{label}-{secrets.token_urlsafe(32)}"


def _settings(tmp_path: Path, service_key: str, admin_key: str) -> Settings:
    return Settings(
        sqlite_path=str(tmp_path / "atagia-sillytavern.db"),
        migrations_path=str(PACKAGE_RESOURCES / "migrations"),
        manifests_path=str(PACKAGE_RESOURCES / "manifests"),
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        openai_api_key="test-openai-key",
        openrouter_api_key=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
        llm_chat_model="openai/test-model",
        llm_ingest_model="openai/test-model",
        llm_retrieval_model="openai/test-model",
        llm_component_models={"intent_classifier": "openai/test-model"},
        service_mode=True,
        service_api_key=service_key,
        admin_api_key=admin_key,
        workers_enabled=False,
        debug=False,
        allow_insecure_http=True,
        small_corpus_token_threshold_ratio=0.0,
    )


@dataclass
class RunningAsgiServer:
    base_url: str
    server: uvicorn.Server
    thread: threading.Thread


@contextmanager
def _run_atagia(settings: Settings) -> Iterator[RunningAsgiServer]:
    app = create_app(settings)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        access_log=False,
        log_level="critical",
        lifespan="on",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(
        target=server.run,
        kwargs={"sockets": [listener]},
        name="atagia-sillytavern-test-asgi",
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=5)
        raise RuntimeError("Atagia ASGI fixture did not start")
    try:
        yield RunningAsgiServer(
            base_url=f"http://127.0.0.1:{port}",
            server=server,
            thread=thread,
        )
    finally:
        server.should_exit = True
        thread.join(timeout=15)
        if thread.is_alive():
            server.force_exit = True
            thread.join(timeout=5)
        listener.close()
        if thread.is_alive():
            raise RuntimeError("Atagia ASGI fixture did not stop")


@dataclass
class RunningNodeBoundary:
    process: subprocess.Popen[str]
    base_url: str
    startup_lines: list[str]
    logs: str = field(default="", init=False)


@contextmanager
def _run_node_boundary(
    node: str,
    *,
    atagia_base_url: str,
    service_key: str,
    user_map: dict[str, str],
) -> Iterator[RunningNodeBoundary]:
    env = os.environ.copy()
    env.update(
        {
            "ATAGIA_BASE_URL": atagia_base_url,
            "ATAGIA_SERVICE_API_KEY": service_key,
            "ATAGIA_SILLYTAVERN_INSTALLATION_ID": "sillytavern-test-installation",
            "ATAGIA_SILLYTAVERN_USER_MAP": json.dumps(user_map),
            "ATAGIA_SILLYTAVERN_TIMEOUT_MS": "5000",
            "ATAGIA_TEST_SILLYTAVERN_HANDLE": "alice",
            "ATAGIA_TEST_CSRF_TOKEN": "test-csrf-token",
        }
    )
    process = subprocess.Popen(
        [node, str(NODE_HARNESS), str(SERVER_PLUGIN)],
        cwd=ROOT,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    startup_lines: list[str] = []
    port: int | None = None
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        ready, _, _ = select.select([process.stdout], [], [], 0.1)
        if not ready:
            continue
        line = process.stdout.readline()
        if not line:
            continue
        startup_lines.append(line)
        if line.startswith("ATAGIA_TEST_SERVER_PORT="):
            port = int(line.partition("=")[2])
            break
    if port is None:
        stdout, stderr = process.communicate(timeout=5)
        raise RuntimeError(
            "SillyTavern boundary fixture did not start:\n"
            + "".join(startup_lines)
            + stdout
            + stderr
        )
    running = RunningNodeBoundary(
        process=process,
        base_url=f"http://127.0.0.1:{port}",
        startup_lines=startup_lines,
    )
    try:
        yield running
    finally:
        if process.poll() is None and process.stdin is not None:
            process.stdin.write("stop\n")
            process.stdin.flush()
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            stdout, stderr = process.communicate(timeout=5)
        running.logs = "".join(startup_lines) + stdout + stderr
        if process.returncode != 0:
            raise RuntimeError(f"SillyTavern boundary fixture failed:\n{running.logs}")


def _service_headers(service_key: str, user_id: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {service_key}",
        "X-Atagia-User-Id": user_id,
        "X-Atagia-Platform-Id": "sillytavern",
    }


def _browser_headers() -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Sec-Fetch-Site": "same-origin",
        "X-CSRF-Token": "test-csrf-token",
    }


def _assert_material_absent_from_repository(materials: tuple[str, ...]) -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout.split(b"\0")
    paths = {ROOT / raw.decode("utf-8") for raw in tracked if raw}
    paths.update(
        path
        for path in (ROOT / "integrations" / "sillytavern").rglob("*")
        if path.is_file()
    )
    encoded = tuple(material.encode("utf-8") for material in materials)
    for path in paths:
        if not path.is_file():
            continue
        contents = path.read_bytes()
        for material in encoded:
            assert material not in contents, f"credential material found in {path}"


def test_old_key_rejected_and_new_same_origin_route_survives_both_restarts(
    tmp_path: Path,
) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")

    compromised_key_copy = _credential("legacy-browser-copy")
    replacement_key = _credential("server-replacement")
    admin_key = _credential("admin")
    mapped_user = "atagia-alice"
    settings = _settings(tmp_path, replacement_key, admin_key)
    node_runs: list[RunningNodeBoundary] = []

    for restart_index in range(2):
        with _run_atagia(settings) as atagia:
            with httpx.Client(timeout=10) as client:
                rejected = client.get(
                    f"{atagia.base_url}/v1/models",
                    headers=_service_headers(compromised_key_copy, mapped_user),
                )
                assert rejected.status_code == 401

                with _run_node_boundary(
                    node,
                    atagia_base_url=atagia.base_url,
                    service_key=replacement_key,
                    user_map={"alice": mapped_user, "bob": "atagia-bob"},
                ) as boundary:
                    node_runs.append(boundary)
                    spoofed = client.post(
                        f"{boundary.base_url}/api/plugins/atagia-memory/health",
                        headers=_browser_headers(),
                        json={"user_id": "atagia-bob"},
                    )
                    assert spoofed.status_code == 400
                    assert spoofed.json() == {
                        "error": "server_managed_identity_required"
                    }
                    health = client.post(
                        f"{boundary.base_url}/api/plugins/atagia-memory/health",
                        headers=_browser_headers(),
                        json={},
                    )
                    assert health.status_code == 200, health.text
                    assert health.json() == {"status": "ok"}
                    assert mapped_user not in health.text

    expected_fingerprint = (
        "sha256:" + hashlib.sha256(replacement_key.encode("utf-8")).hexdigest()[:16]
    )
    assert len(node_runs) == 2
    for boundary in node_runs:
        assert expected_fingerprint in boundary.logs
        assert compromised_key_copy not in boundary.logs
        assert replacement_key not in boundary.logs
        assert admin_key not in boundary.logs

    browser_visible_trace = json.dumps(
        {
            "request_headers": _browser_headers(),
            "response_status": health.status_code,
            "response_body": health.json(),
        }
    )
    for material in (compromised_key_copy, replacement_key, admin_key):
        assert material not in browser_visible_trace
    _assert_material_absent_from_repository(
        (compromised_key_copy, replacement_key, admin_key)
    )


def test_declared_sillytavern_host_pin_matches_local_checkout() -> None:
    if not (SILLYTAVERN_CHECKOUT / ".git").exists():
        pytest.skip("the pinned local SillyTavern checkout is unavailable")
    package = json.loads(
        (SILLYTAVERN_CHECKOUT / "package.json").read_text(encoding="utf-8")
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=SILLYTAVERN_CHECKOUT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    assert package["version"] == SILLYTAVERN_VERSION
    assert commit == SILLYTAVERN_COMMIT

    loader_source = (SILLYTAVERN_CHECKOUT / "src" / "plugin-loader.js").read_text(
        encoding="utf-8"
    )
    assert "const info = plugin.info || plugin.default?.info" in loader_source
    assert "const init = plugin.init || plugin.default?.init" in loader_source
    assert "app.use(`/api/plugins/${id}`, router)" in loader_source

    extension_loader_source = (
        SILLYTAVERN_CHECKOUT / "public" / "scripts" / "extensions.js"
    ).read_text(encoding="utf-8")
    assert "return callExtensionHook(name, 'activate');" in extension_loader_source
    assert (
        "globalThis[interceptorKey](chat, contextSize, abort, type)"
        in extension_loader_source
    )
    client_source = (SILLYTAVERN_CHECKOUT / "public" / "script.js").read_text(
        encoding="utf-8"
    )
    assert "'X-CSRF-Token': token" in client_source
    assert (
        "eventSource.emit(event_types.MESSAGE_RECEIVED, this.messageId, this.type)"
        in client_source
    )
    assert (
        "eventSource.emit(event_types.MESSAGE_RECEIVED, chat_id, type)" in client_source
    )
