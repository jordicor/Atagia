"""Repository helpers for aggregated evaluation metrics."""

from __future__ import annotations

from typing import Any

from atagia.core.ids import generate_prefixed_id
from atagia.core.repositories import BaseRepository


class MetricsRepository(BaseRepository):
    """CRUD operations for the evaluation_metrics aggregate table."""

    async def store_metric(
        self,
        *,
        metric_name: str,
        value: float,
        sample_count: int,
        time_bucket: str,
        computed_at: str | None = None,
        user_id: str | None = None,
        assistant_mode_id: str | None = None,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        metric_id = generate_prefixed_id("met")
        timestamp = computed_at or self._timestamp()
        await self._connection.execute(
            """
            INSERT OR REPLACE INTO evaluation_metrics(
                id,
                metric_name,
                metric_value,
                sample_count,
                user_id,
                assistant_mode_id,
                workspace_id,
                time_bucket,
                computed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                metric_id,
                metric_name,
                value,
                sample_count,
                user_id,
                assistant_mode_id,
                workspace_id,
                time_bucket,
                timestamp,
            ),
        )
        await self._connection.commit()
        stored = await self.get_metric(
            metric_name=metric_name,
            time_bucket=time_bucket,
            user_id=user_id,
            assistant_mode_id=assistant_mode_id,
        )
        if stored is None:
            raise RuntimeError(f"Failed to store metric {metric_name} for bucket {time_bucket}")
        return stored

    async def get_metric(
        self,
        *,
        metric_name: str,
        time_bucket: str,
        user_id: str | None = None,
        assistant_mode_id: str | None = None,
    ) -> dict[str, Any] | None:
        return await self._fetch_one(
            """
            SELECT *
            FROM evaluation_metrics
            WHERE metric_name = ?
              AND time_bucket = ?
              AND user_id IS ?
              AND assistant_mode_id IS ?
            ORDER BY computed_at DESC, id DESC
            LIMIT 1
            """,
            (metric_name, time_bucket, user_id, assistant_mode_id),
        )

    async def list_metrics(
        self,
        *,
        metric_name: str,
        user_id: str | None = None,
        assistant_mode_id: str | None = None,
        from_bucket: str | None = None,
        to_bucket: str | None = None,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        clauses = ["metric_name = ?", "user_id IS ?", "assistant_mode_id IS ?"]
        parameters: list[Any] = [metric_name, user_id, assistant_mode_id]
        if from_bucket is not None:
            clauses.append("time_bucket >= ?")
            parameters.append(from_bucket)
        if to_bucket is not None:
            clauses.append("time_bucket <= ?")
            parameters.append(to_bucket)
        parameters.append(limit)
        return await self._fetch_all(
            """
            SELECT *
            FROM evaluation_metrics
            WHERE {where_clause}
            ORDER BY time_bucket DESC, computed_at DESC, id DESC
            LIMIT ?
            """.format(where_clause=" AND ".join(clauses)),
            tuple(parameters),
        )

    async def get_latest_metrics(
        self,
        *,
        user_id: str | None = None,
        assistant_mode_id: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        rows = await self._fetch_all(
            """
            SELECT
                id,
                metric_name,
                metric_value,
                sample_count,
                user_id,
                assistant_mode_id,
                workspace_id,
                time_bucket,
                computed_at
            FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY metric_name
                        ORDER BY time_bucket DESC, computed_at DESC, id DESC
                    ) AS metric_rank
                FROM evaluation_metrics
                WHERE user_id IS ?
                  AND assistant_mode_id IS ?
            )
            WHERE metric_rank = 1
            ORDER BY metric_name ASC
            """,
            (user_id, assistant_mode_id),
        )
        return {str(row["metric_name"]): row for row in rows}

    async def delete_old_metrics(self, *, older_than_bucket: str) -> int:
        cursor = await self._connection.execute(
            """
            DELETE FROM evaluation_metrics
            WHERE time_bucket < ?
            """,
            (older_than_bucket,),
        )
        await self._connection.commit()
        return int(cursor.rowcount or 0)
