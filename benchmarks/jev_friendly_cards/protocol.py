"""Build and verify independent A/B/C evaluation slots without dispatching models."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any


ARMS = ("A_baseline_llm", "B_shared_llm", "C_shared_jev")
REPETITION_OPTIONS = frozenset({3, 5, 10})


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process_git_environment(*roots: Path) -> dict[str, str]:
    """Trust only the selected checkouts for child Git calls in this process."""
    env = dict(os.environ)
    env["GIT_CONFIG_COUNT"] = str(len(roots))
    for index, root in enumerate(roots):
        env[f"GIT_CONFIG_KEY_{index}"] = "safe.directory"
        env[f"GIT_CONFIG_VALUE_{index}"] = root.resolve(strict=True).as_posix()
    return env


def imported_root(
    root: Path, python: Path, harness_root: Path | None = None
) -> dict[str, str]:
    """Prove production and benchmark imports come from their selected checkouts."""
    root = root.resolve(strict=True)
    python = python.resolve(strict=True)
    harness_root = (harness_root or root).resolve(strict=True)
    code = (
        "import json, pathlib, atagia, benchmarks; "
        "print(json.dumps({'atagia': str(pathlib.Path(atagia.__file__).resolve()), "
        "'benchmarks': str(pathlib.Path(benchmarks.__file__).resolve())}))"
    )
    env = process_git_environment(root, harness_root)
    env["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(harness_root)))
    output = subprocess.check_output(
        [str(python), "-c", code], cwd=harness_root, env=env, text=True
    )
    paths = json.loads(output)
    if not Path(paths["atagia"]).is_relative_to(root / "src" / "atagia"):
        raise ValueError("Atagia import escaped the selected worktree")
    if not Path(paths["benchmarks"]).is_relative_to(harness_root / "benchmarks"):
        raise ValueError("Benchmark import escaped the selected worktree")
    return paths


def root_fingerprint(
    root: Path, python: Path, harness_root: Path | None = None
) -> dict[str, Any]:
    """Bind a variant to one Git checkout and a superset of production sources."""
    root = root.resolve(strict=True)
    imports = imported_root(root, python, harness_root)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        env=process_git_environment(root),
        text=True,
    ).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Selected worktree has no exact Git revision")
    sources = sorted((root / "src" / "atagia").rglob("*.py"))
    if not sources:
        raise ValueError("Production source tree is empty")
    return {
        "root": str(root),
        "revision": revision,
        "imports": imports,
        "source_hashes": {path.relative_to(root).as_posix(): sha256(path) for path in sources},
    }


def plan_slots(cases: list[dict[str, Any]], repetitions: int) -> list[dict[str, Any]]:
    """Choose equal repetitions and balanced order before any evaluation output."""
    if repetitions not in REPETITION_OPTIONS:
        raise ValueError("Comparable groups require 10, 5, or 3 repetitions")
    identifiers = [case["case_id"] for case in cases]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Case IDs must be unique")
    slots = []
    for repetition in range(1, repetitions + 1):
        for index, case in enumerate(cases):
            offset = (index + repetition - 1) % len(ARMS)
            for arm in ARMS[offset:] + ARMS[:offset]:
                slots.append(
                    {
                        "slot": f"evaluation:{case['case_id']}:{arm}:{repetition:02}",
                        "case_id": case["case_id"],
                        "arm": arm,
                        "repetition": repetition,
                        "family": case["primary_family"],
                    }
                )
    return slots


def terminal_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        slot = row["slot"]
        if slot in rows:
            raise ValueError(f"Duplicate terminal slot: {slot}")
        if row["status"] not in {
            "success", "invalid_response", "provider_error", "harness_error"
        }:
            raise ValueError(f"Unknown terminal status: {row['status']}")
        rows[slot] = row
    return rows


def remaining_slots(
    planned: list[dict[str, Any]],
    terminal: dict[str, dict[str, Any]],
    attempted_slots: set[str],
    pending_reservations: set[str],
) -> list[dict[str, Any]]:
    """Refuse resume with an unrecorded attempt or unresolved reservation."""
    planned_ids = {slot["slot"] for slot in planned}
    if len(planned_ids) != len(planned) or set(terminal) - planned_ids:
        raise ValueError("Unknown or repeated planned slot")
    if pending_reservations or attempted_slots - set(terminal):
        raise ValueError("Paid attempts must be reconciled before resume")
    return [slot for slot in planned if slot["slot"] not in terminal]
