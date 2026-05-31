"""CLI entry point for content-language metadata backfill."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from typing import Sequence

from atagia.app import initialize_runtime
from atagia.core.config import Settings
from atagia.services.content_language_backfill_service import (
    ContentLanguageBackfillService,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the content-language backfill CLI."""
    parser = argparse.ArgumentParser(
        prog="atagia-content-language-backfill",
        description="Classify and backfill missing memory content-language metadata.",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--delay-ms", type=int, default=0)
    parser.add_argument("--user-id", type=str, default=None)
    parser.add_argument("--min-confidence", type=float, default=0.45)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write language_codes_json updates. Default is dry-run only.",
    )
    return parser


async def main_async(argv: Sequence[str] | None = None) -> int:
    """Run the content-language backfill CLI."""
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
        result = await ContentLanguageBackfillService(
            connection=connection,
            llm_client=runtime.llm_client,
            settings=runtime.settings,
        ).run(
            batch_size=args.batch_size,
            delay_ms=args.delay_ms,
            user_id=args.user_id,
            dry_run=not args.write,
            min_confidence=args.min_confidence,
        )
    finally:
        await connection.close()
        await runtime.close()
    print(result.model_dump_json())
    return 0


def main() -> None:
    """System-exit wrapper for setuptools scripts."""
    raise SystemExit(asyncio.run(main_async()))
