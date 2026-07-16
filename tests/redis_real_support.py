"""Shared lifecycle for isolated real-Redis integration tests."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import time
from typing import Iterator
from uuid import uuid4

import pytest
from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError


@dataclass(frozen=True, slots=True)
class RealRedisServer:
    """One isolated Redis server owned by a test."""

    url: str
    client: Redis


@contextmanager
def running_redis_server(tmp_path: Path) -> Iterator[RealRedisServer]:
    """Start Redis without relying on an already-running developer service."""

    executable = shutil.which("redis-server")
    if executable is None:
        pytest.skip("redis-server is required for the real Redis integration gate")
    socket_path = Path("/tmp") / f"atagia-test-{uuid4().hex}.sock"
    process = subprocess.Popen(
        [
            executable,
            "--save",
            "",
            "--appendonly",
            "no",
            "--port",
            "0",
            "--unixsocket",
            str(socket_path),
            "--unixsocketperm",
            "700",
            "--dir",
            str(tmp_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    client = Redis(unix_socket_path=str(socket_path), decode_responses=True)
    deadline = time.monotonic() + 5.0
    while True:
        if process.poll() is not None:
            raise RuntimeError(
                f"redis-server exited during startup ({process.returncode})"
            )
        try:
            if client.ping():
                break
        except (RedisConnectionError, FileNotFoundError):
            pass
        if time.monotonic() >= deadline:
            process.kill()
            process.wait(timeout=5)
            raise RuntimeError("redis-server did not become ready")
        time.sleep(0.01)

    try:
        client.flushall()
        yield RealRedisServer(
            url=f"unix://{socket_path}",
            client=client,
        )
    finally:
        try:
            client.shutdown(nosave=True)
        except (RedisConnectionError, OSError):
            pass
        client.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        socket_path.unlink(missing_ok=True)
