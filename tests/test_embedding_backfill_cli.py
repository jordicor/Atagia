"""Tests for the embedding backfill CLI."""

from __future__ import annotations

import pytest

from atagia.embedding_backfill_cli import build_parser, main_async
from atagia.services.embedding_backfill_service import EmbeddingBackfillResult


def test_build_parser_accepts_backfill_flags() -> None:
    args = build_parser().parse_args(["--batch-size", "25", "--delay-ms", "50", "--user-id", "usr_9"])

    assert args.batch_size == 25
    assert args.delay_ms == 50
    assert args.user_id == "usr_9"


@pytest.mark.asyncio
async def test_main_async_runs_backfill_and_prints_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeConnection:
        async def close(self) -> None:
            return None

    class FakeRuntime:
        def __init__(self) -> None:
            self.embedding_index = object()

        async def open_connection(self) -> FakeConnection:
            return FakeConnection()

        async def close(self) -> None:
            return None

    captured: dict[str, object] = {}

    class FakeBackfillService:
        def __init__(self, *, connection, embedding_index) -> None:
            captured["connection"] = connection
            captured["embedding_index"] = embedding_index

        async def run(self, *, batch_size: int, delay_ms: int, user_id: str | None) -> EmbeddingBackfillResult:
            captured["batch_size"] = batch_size
            captured["delay_ms"] = delay_ms
            captured["user_id"] = user_id
            return EmbeddingBackfillResult(
                examined=3,
                embedded=2,
                skipped=1,
                failed=0,
                batch_size=batch_size,
                delay_ms=delay_ms,
                user_id=user_id,
            )

    async def _fake_initialize_runtime(settings):
        captured["workers_enabled"] = settings.workers_enabled
        captured["lifecycle_worker_enabled"] = settings.lifecycle_worker_enabled
        return FakeRuntime()

    monkeypatch.setattr("atagia.embedding_backfill_cli.initialize_runtime", _fake_initialize_runtime)
    monkeypatch.setattr("atagia.embedding_backfill_cli.EmbeddingBackfillService", FakeBackfillService)

    exit_code = await main_async(["--batch-size", "25", "--delay-ms", "50", "--user-id", "usr_9"])

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert captured["batch_size"] == 25
    assert captured["delay_ms"] == 50
    assert captured["user_id"] == "usr_9"
    assert captured["workers_enabled"] is False
    assert captured["lifecycle_worker_enabled"] is False
    assert '"embedded":2' in stdout
