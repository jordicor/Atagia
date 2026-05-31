"""Lifecycle runner coordination tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.db_sqlite import close_connection, initialize_database, open_connection
from atagia.core.storage_backend import InProcessBackend
from atagia.memory.lifecycle_runner import (
    _runtime_prefix,
    piggyback_lifecycle,
    try_run_lifecycle,
)
from atagia.services.embeddings import NoneBackend

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


def _settings(db_path: Path) -> Settings:
    return Settings(
        sqlite_path=str(db_path),
        migrations_path=str(MIGRATIONS_DIR),
        manifests_path=str(MANIFESTS_DIR),
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        openai_api_key="test-openai-key",
        openrouter_api_key=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
        llm_chat_model="reply-test-model",
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        lifecycle_busy_timeout_ms=1,
        lifecycle_busy_backoff_seconds=60,
        lifecycle_failure_backoff_seconds=300,
        debug=False,
    )


@dataclass(slots=True)
class _RuntimeView:
    database_path: str
    clock: FrozenClock
    settings: Settings
    embedding_index: NoneBackend
    storage_backend: InProcessBackend
    artifact_blob_store: None = None
    llm_client: None = None


async def _initialize_file_db(db_path: Path) -> None:
    connection = await initialize_database(str(db_path), MIGRATIONS_DIR)
    await close_connection(connection)


@pytest.mark.asyncio
async def test_piggyback_sqlite_busy_sets_short_backoff_without_error_log(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    db_path = tmp_path / "atagia.db"
    await _initialize_file_db(db_path)
    settings = _settings(db_path)
    storage = InProcessBackend()
    runtime = _RuntimeView(
        database_path=str(db_path),
        clock=FrozenClock(datetime(2026, 5, 21, 12, 0, tzinfo=timezone.utc)),
        settings=settings,
        embedding_index=NoneBackend(),
        storage_backend=storage,
    )

    writer = await open_connection(str(db_path))
    try:
        await writer.execute("BEGIN IMMEDIATE")
        caplog.set_level(logging.ERROR, logger="atagia.memory.lifecycle_runner")

        await piggyback_lifecycle(runtime)

        prefix = _runtime_prefix(str(db_path))
        assert not await storage.has_dedupe(f"lifecycle:cooldown:{prefix}")
        assert await storage.has_dedupe(f"lifecycle:busy_backoff:{prefix}")
        assert "Piggyback lifecycle failed" not in caplog.text
    finally:
        await writer.rollback()
        await close_connection(writer)


@pytest.mark.asyncio
async def test_successful_lifecycle_sets_success_cooldown(tmp_path: Path) -> None:
    db_path = tmp_path / "atagia.db"
    await _initialize_file_db(db_path)
    settings = _settings(db_path)
    storage = InProcessBackend()
    clock = FrozenClock(datetime(2026, 5, 21, 12, 0, tzinfo=timezone.utc))

    first_result = await try_run_lifecycle(
        database_path=str(db_path),
        clock=clock,
        settings=settings,
        embedding_index=NoneBackend(),
        storage_backend=storage,
    )
    second_result = await try_run_lifecycle(
        database_path=str(db_path),
        clock=clock,
        settings=settings,
        embedding_index=NoneBackend(),
        storage_backend=storage,
    )

    prefix = _runtime_prefix(str(db_path))
    assert first_result is True
    assert second_result is False
    assert await storage.has_dedupe(f"lifecycle:cooldown:{prefix}")
