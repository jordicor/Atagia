"""Benchmark tests.

Expose the repository-level ``benchmarks`` package during pytest collection.
"""

from pathlib import Path

_PROJECT_BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"
__path__ = [str(_PROJECT_BENCHMARKS), *list(__path__)]
