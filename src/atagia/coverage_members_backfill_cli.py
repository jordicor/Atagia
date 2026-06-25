"""CLI entry point for coverage-members payload backfill."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from typing import Sequence

from atagia.app import initialize_runtime
from atagia.core.config import Settings
from atagia.services.coverage_members_backfill_service import (
    CoverageMembersBackfillService,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the coverage-members backfill CLI."""
    parser = argparse.ArgumentParser(
        prog="atagia-coverage-members-backfill",
        description="Re-derive and backfill missing memory coverage_members metadata.",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--delay-ms", type=int, default=0)
    parser.add_argument("--user-id", type=str, default=None)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write coverage_members updates. Default is dry-run only.",
    )
    return parser


async def main_async(argv: Sequence[str] | None = None) -> int:
    """Run the coverage-members backfill CLI."""
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
        result = await CoverageMembersBackfillService(
            connection=connection,
            llm_client=runtime.llm_client,
            settings=runtime.settings,
        ).run(
            batch_size=args.batch_size,
            delay_ms=args.delay_ms,
            user_id=args.user_id,
            dry_run=not args.write,
        )
    finally:
        await connection.close()
        await runtime.close()
    print(result.model_dump_json())
    return 0


def main() -> None:
    """System-exit wrapper for setuptools scripts."""
    raise SystemExit(asyncio.run(main_async()))
