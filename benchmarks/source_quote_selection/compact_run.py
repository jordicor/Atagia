"""Compare compact native selectors through the unchanged evidence workflow."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from unittest.mock import patch

from benchmarks.source_quote_selection import workflow
from benchmarks.source_quote_selection.run import ROOT, digest, write_json

ASSIGNMENT = "20260924-source-quote-compaction-v1"
REPETITIONS = 3
ARMS = {
    **workflow.ARMS,
    "jev_compact": workflow.ARMS["jev_with_luna6"],
    "jev_focused": workflow.ARMS["jev_with_luna6"],
}
SOURCES = (
    *workflow.SOURCES,
    "benchmarks/source_quote_selection/compact_selector.py",
    "benchmarks/source_quote_selection/compact_cases.py",
    "benchmarks/source_quote_selection/compact_run.py",
)
_PRODUCTION_EVALUATE = workflow.evaluate


async def evaluate(client, case: dict, arm: str) -> dict:
    """Replace only the experimental selector, inside one sequential operation."""
    if arm not in ARMS:
        raise ValueError(f"Unknown compaction arm: {arm}")
    if arm in {"jev_compact", "jev_focused"}:
        from benchmarks.source_quote_selection.compact_selector import (
            select_source_references,
        )

        selector = partial(select_source_references, mode=arm.removeprefix("jev_"))
        # This patch exists only in the single-owner benchmark process. Production
        # runs the ordinary helper unchanged, including annotation cost and order.
        with patch(
            "atagia.memory.source_quote_selector.select_source_references", selector
        ):
            return await _PRODUCTION_EVALUATE(client, case, "jev_with_luna6")
    return await _PRODUCTION_EVALUATE(client, case, arm)


@contextmanager
def _run_configuration():
    """Reuse the existing capture/reservation runner without changing its file."""
    with patch.multiple(
        workflow,
        ASSIGNMENT=ASSIGNMENT,
        ARMS=ARMS,
        SOURCES=SOURCES,
        evaluate=evaluate,
    ):
        yield


def prepare(output: Path, helper: Path) -> None:
    from benchmarks.source_quote_selection.compact_cases import load_cases

    if output.exists():
        raise ValueError("Never overwrite a compaction experiment")
    output.mkdir(parents=True)
    phases = {"smoke": workflow.smoke_cases(), "acceptance": load_cases()}
    slots = []
    for phase, cases in phases.items():
        repetitions = 1 if phase == "smoke" else REPETITIONS
        for repetition in range(1, repetitions + 1):
            for index, case in enumerate(cases):
                arms = list(ARMS)
                offset = (index + repetition) % len(arms)
                for arm in arms[offset:] + arms[:offset]:
                    slots.append(
                        {
                            "slot": f"{phase}:{case['case_id']}:{arm}:{repetition}",
                            "phase": phase,
                            "case_id": case["case_id"],
                            "arm": arm,
                            "repetition": repetition,
                        }
                    )
    write_json(
        output / "manifest.json",
        {
            "arms": ARMS,
            "cases": phases,
            "slots": slots,
            "acceptance_repetitions": REPETITIONS,
            "scope": "Complete evidence operation; fixed candidates, no cross-job waits",
            "variants": {
                "jev_compact": "Null reference descriptions; unchanged blocks and state",
                "jev_focused": "Compact options, long-source blocks of 64, selected interval union",
            },
        },
    )
    write_json(
        output / "freeze.json",
        {
            "sources": {path: digest(ROOT / path) for path in SOURCES},
            "helper": digest(helper),
            "manifest": digest(output / "manifest.json"),
        },
    )


def verify(output: Path, helper: Path) -> dict:
    with _run_configuration():
        return workflow.verify(output, helper)


async def run(output: Path, helper: Path, phase: str) -> None:
    with _run_configuration():
        await workflow.run(output, helper, phase)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--budget-helper", required=True, type=Path)
    parser.add_argument("--phase", choices=("smoke", "acceptance"))
    args = parser.parse_args()
    if args.phase:
        asyncio.run(run(args.output, args.budget_helper, args.phase))
    else:
        prepare(args.output, args.budget_helper)


if __name__ == "__main__":
    main()
