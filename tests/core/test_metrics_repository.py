"""Tests for aggregated evaluation metric storage."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from atagia.core.clock import FrozenClock
from atagia.core.db_sqlite import initialize_database
from atagia.core.metrics_repository import MetricsRepository

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


async def _build_runtime() -> tuple[object, FrozenClock, MetricsRepository]:
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 3, 31, 12, 0, tzinfo=timezone.utc))
    repository = MetricsRepository(connection, clock)
    return connection, clock, repository


@pytest.mark.asyncio
async def test_store_metric_creates_row() -> None:
    connection, _clock, repository = await _build_runtime()
    try:
        row = await repository.store_metric(
            metric_name="mur",
            value=0.5,
            sample_count=4,
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            time_bucket="2026-03-31",
        )

        assert row["metric_name"] == "mur"
        assert row["metric_value"] == pytest.approx(0.5)
        assert row["sample_count"] == 4
        assert row["user_id"] == "usr_1"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_get_metric_returns_by_name_and_bucket() -> None:
    connection, _clock, repository = await _build_runtime()
    try:
        await repository.store_metric(
            metric_name="mur",
            value=0.25,
            sample_count=4,
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            time_bucket="2026-03-30",
        )
        await repository.store_metric(
            metric_name="mur",
            value=0.75,
            sample_count=8,
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            time_bucket="2026-03-31",
        )

        row = await repository.get_metric(
            metric_name="mur",
            time_bucket="2026-03-31",
            user_id="usr_1",
            assistant_mode_id="coding_debug",
        )

        assert row is not None
        assert row["metric_value"] == pytest.approx(0.75)
        assert row["sample_count"] == 8
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_list_metrics_returns_ordered_desc_and_respects_range() -> None:
    connection, _clock, repository = await _build_runtime()
    try:
        for bucket, value in (
            ("2026-03-29", 0.1),
            ("2026-03-30", 0.2),
            ("2026-03-31", 0.3),
        ):
            await repository.store_metric(
                metric_name="ipr",
                value=value,
                sample_count=5,
                user_id="usr_1",
                assistant_mode_id="coding_debug",
                time_bucket=bucket,
            )

        rows = await repository.list_metrics(
            metric_name="ipr",
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            from_bucket="2026-03-30",
            to_bucket="2026-03-31",
            limit=10,
        )

        assert [row["time_bucket"] for row in rows] == ["2026-03-31", "2026-03-30"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_get_latest_metrics_returns_most_recent_per_metric_name() -> None:
    connection, _clock, repository = await _build_runtime()
    try:
        await repository.store_metric(
            metric_name="mur",
            value=0.4,
            sample_count=5,
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            time_bucket="2026-03-30",
        )
        await repository.store_metric(
            metric_name="mur",
            value=0.9,
            sample_count=6,
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            time_bucket="2026-03-31",
        )
        await repository.store_metric(
            metric_name="ipr",
            value=0.2,
            sample_count=6,
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            time_bucket="2026-03-31",
        )

        latest = await repository.get_latest_metrics(
            user_id="usr_1",
            assistant_mode_id="coding_debug",
        )

        assert latest["mur"]["metric_value"] == pytest.approx(0.9)
        assert latest["ipr"]["metric_value"] == pytest.approx(0.2)
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_delete_old_metrics_removes_old_entries() -> None:
    connection, _clock, repository = await _build_runtime()
    try:
        await repository.store_metric(
            metric_name="mur",
            value=0.4,
            sample_count=5,
            time_bucket="2026-03-29",
        )
        await repository.store_metric(
            metric_name="mur",
            value=0.8,
            sample_count=5,
            time_bucket="2026-03-31",
        )

        deleted = await repository.delete_old_metrics(older_than_bucket="2026-03-30")
        rows = await repository.list_metrics(metric_name="mur", limit=10)

        assert deleted == 1
        assert [row["time_bucket"] for row in rows] == ["2026-03-31"]
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_store_metric_replaces_existing_bucket_row() -> None:
    connection, _clock, repository = await _build_runtime()
    try:
        first = await repository.store_metric(
            metric_name="mur",
            value=0.4,
            sample_count=5,
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            time_bucket="2026-03-31",
        )
        second = await repository.store_metric(
            metric_name="mur",
            value=0.8,
            sample_count=9,
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            time_bucket="2026-03-31",
        )
        rows = await repository.list_metrics(
            metric_name="mur",
            user_id="usr_1",
            assistant_mode_id="coding_debug",
            limit=10,
        )

        assert len(rows) == 1
        assert rows[0]["metric_value"] == pytest.approx(0.8)
        assert first["id"] != second["id"]
    finally:
        await connection.close()
