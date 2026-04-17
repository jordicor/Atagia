"""CLI entry point for embedding backfill."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from typing import Sequence

from atagia.app import initialize_runtime
from atagia.core.config import Settings
from atagia.services.embedding_backfill_service import EmbeddingBackfillService


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the backfill CLI."""
    parser = argparse.ArgumentParser(
        prog="atagia-embeddings-backfill",
        description="Backfill sqlite-vec embeddings for existing memory rows.",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--delay-ms", type=int, default=0)
    parser.add_argument("--user-id", type=str, default=None)
    return parser


async def main_async(argv: Sequence[str] | None = None) -> int:
    """Run the embedding backfill CLI."""
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    runtime = await initialize_runtime(
        replace(
            Settings.from_env(),
            workers_enabled=False,
            lifecycle_worker_enabled=False,
        )
    )
    connection = await runtime.open_connection()
    try:
        result = await EmbeddingBackfillService(
            connection=connection,
            embedding_index=runtime.embedding_index,
        ).run(
            batch_size=args.batch_size,
            delay_ms=args.delay_ms,
            user_id=args.user_id,
        )
    finally:
        await connection.close()
        await runtime.close()
    print(result.model_dump_json())
    return 0


def main() -> None:
    """System-exit wrapper for setuptools scripts."""
    raise SystemExit(asyncio.run(main_async()))
