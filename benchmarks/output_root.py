"""Shared guardrail for benchmark output locations.

Benchmark runs emit large artifacts (reports, DB snapshots, checkpoints) that
must never land inside the Atagia repository -- committing them is exactly what
bloated the repo before. They belong in a sibling output root (default
``../atagia-benchmarks``), overridable with ``$ATAGIA_BENCH_OUTPUT_ROOT``.

Every runner resolves its output through this module so that:

1. a path resolving inside the repo fails fast (no silent re-pollution), and
2. each run carries a UTC component, so runs never overwrite one another.

The UTC recognizer here is mechanical/structural parsing of a known timestamp
grammar (``YYYYMMDD`` optionally followed by ``THHMMSSZ``), not a semantic task,
so a regex is appropriate.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path

ENV_OUTPUT_ROOT = "ATAGIA_BENCH_OUTPUT_ROOT"

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_ROOT = _REPO_ROOT.parent / "atagia-benchmarks"

# Matches a single path component that is a UTC stamp such as ``20260613`` or
# ``20260613T170000Z``. Anchored: only whole-component stamps count.
_UTC_COMPONENT = re.compile(r"^\d{8}(?:T\d{6}Z)?$")


def repo_root() -> Path:
    """Return the resolved Atagia repository root."""
    return _REPO_ROOT


def default_output_root() -> Path:
    """Return the default external output root (``../atagia-benchmarks``)."""
    return _DEFAULT_ROOT


def utc_run_id(now: datetime | None = None) -> str:
    """Return a filesystem-safe UTC run id such as ``20260613T170000Z``."""
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def is_inside_repo(path: str | Path) -> bool:
    """Return whether path resolves to the repo root or anything beneath it."""
    candidate = Path(path).expanduser().resolve(strict=False)
    if candidate == _REPO_ROOT:
        return True
    try:
        candidate.relative_to(_REPO_ROOT)
    except ValueError:
        return False
    return True


def assert_outside_repo(path: str | Path) -> Path:
    """Fail fast if path would place benchmark artifacts inside the repo.

    Returns the resolved path on success.
    """
    resolved = Path(path).expanduser().resolve(strict=False)
    if is_inside_repo(resolved):
        raise ValueError(
            "Refusing to write benchmark artifacts inside the Atagia repo: "
            f"{resolved}. Benchmark output must live outside the repo; set "
            f"${ENV_OUTPUT_ROOT} or write under {_DEFAULT_ROOT}."
        )
    return resolved


def bench_output_root() -> Path:
    """Resolve the benchmark output root, guaranteed outside the repo.

    Uses ``$ATAGIA_BENCH_OUTPUT_ROOT`` when set, otherwise the sibling
    ``../atagia-benchmarks`` directory. Always absolute; fails fast if it
    resolves inside the repo.
    """
    raw = os.environ.get(ENV_OUTPUT_ROOT)
    root = Path(raw).expanduser() if raw else _DEFAULT_ROOT
    return assert_outside_repo(root)


def has_utc_component(path: str | Path, *, relative_to: str | Path | None = None) -> bool:
    """Return whether any path component (below relative_to) is a UTC stamp."""
    resolved = Path(path).expanduser().resolve(strict=False)
    parts: tuple[str, ...]
    if relative_to is not None:
        base = Path(relative_to).expanduser().resolve(strict=False)
        try:
            parts = resolved.relative_to(base).parts
        except ValueError:
            parts = resolved.parts
    else:
        parts = resolved.parts
    return any(_UTC_COMPONENT.match(part) for part in parts)


def resolve_output_dir(
    *parts: str | Path,
    root: str | Path | None = None,
    timestamp: str | None = None,
    require_utc: bool = True,
) -> Path:
    """Resolve a benchmark output directory under the external root.

    ``parts`` are appended under the root (``bench_output_root()`` by default,
    or an explicit external ``root``). When ``require_utc`` is set and none of
    the resulting components below the root looks like a UTC stamp, a ``<UTC>``
    component is appended automatically (auto-stamp), so runs never collide and
    a forgotten timestamp cannot happen. Pass ``timestamp`` for a deterministic
    stamp (tests, or to share one id across sibling artifacts).

    The final path is always asserted to live outside the repo. An absolute
    ``part`` overrides the root (pathlib semantics) but is still boundary-checked.
    """
    base = Path(root).expanduser().resolve(strict=False) if root is not None else bench_output_root()
    target = base
    for part in parts:
        target = target / part
    target = target.resolve(strict=False)
    if require_utc and not has_utc_component(target, relative_to=base):
        target = (target / (timestamp or utc_run_id())).resolve(strict=False)
    return assert_outside_repo(target)
