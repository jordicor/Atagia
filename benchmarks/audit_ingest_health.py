"""Audit retained benchmark DBs for ingestion/provenance trust."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.ingest_health import audit_benchmark_db, classify_ingest_health
from benchmarks.llm_run_guard import LLMRunGuardConfig


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_ROOTS = (
    _PROJECT_ROOT / "benchmarks" / "results",
    _PROJECT_ROOT / "docs" / "tmp" / "benchmark_dbs",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="*",
        default=[str(path) for path in _DEFAULT_ROOTS],
        help="Directories or benchmark.db files to audit.",
    )
    parser.add_argument(
        "--require-evidence-packets",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    return parser


def audit_roots(
    roots: list[str | Path],
    *,
    require_evidence_packets: bool = True,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for db_path in _iter_db_paths(roots):
        metadata = _read_metadata(db_path.parent / "run_metadata.json")
        llm_call_summary = (
            metadata.get("llm_call_summary")
            if isinstance(metadata.get("llm_call_summary"), dict)
            else {}
        )
        health = classify_ingest_health(
            db_audit=audit_benchmark_db(db_path),
            llm_call_summary=llm_call_summary,
            require_evidence_packets=require_evidence_packets,
            llm_guard_config=LLMRunGuardConfig(),
            rebuild_result=(
                metadata.get("rebuild_result")
                if isinstance(metadata.get("rebuild_result"), dict)
                else None
            ),
        )
        entries.append(
            {
                "db_path": str(db_path),
                "metadata_path": str(db_path.parent / "run_metadata.json")
                if (db_path.parent / "run_metadata.json").is_file()
                else None,
                "conversation_id": metadata.get("conversation_id"),
                "status": metadata.get("status"),
                "metadata_trusted_ingest": metadata.get("trusted_ingest"),
                "trusted_ingest": health["trusted_ingest"],
                "reasons": health["reasons"],
                "warnings": health["warnings"],
                "counts": health["db_audit"]["counts"],
                "object_type_counts": health["db_audit"]["object_type_counts"],
            }
        )
    entries.sort(key=lambda item: item["db_path"])
    return {
        "total": len(entries),
        "trusted": sum(1 for entry in entries if entry["trusted_ingest"]),
        "untrusted": sum(1 for entry in entries if not entry["trusted_ingest"]),
        "entries": entries,
    }


def format_audit(summary: dict[str, Any]) -> str:
    lines = [
        "Retained benchmark DB ingest health:",
        (
            f"total={summary['total']} trusted={summary['trusted']} "
            f"untrusted={summary['untrusted']}"
        ),
    ]
    for entry in summary["entries"]:
        status = "trusted" if entry["trusted_ingest"] else "untrusted"
        counts = entry.get("counts") or {}
        lines.append(
            f"- {status} {entry['db_path']} "
            f"memory_objects={counts.get('memory_objects', 0)} "
            f"support_edges={counts.get('memory_support_edges', 0)} "
            f"evidence_spans={counts.get('memory_evidence_spans', 0)}"
        )
        for reason in entry.get("reasons") or []:
            lines.append(f"  reason: {reason}")
    return "\n".join(lines)


def _iter_db_paths(roots: list[str | Path]) -> list[Path]:
    db_paths: set[Path] = set()
    for root in roots:
        path = Path(root).expanduser()
        if path.is_file() and path.name == "benchmark.db":
            db_paths.add(path)
        elif path.is_dir():
            db_paths.update(path.rglob("benchmark.db"))
    return sorted(db_paths)


def _read_metadata(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    args = build_parser().parse_args()
    summary = audit_roots(
        args.roots,
        require_evidence_packets=args.require_evidence_packets,
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(format_audit(summary))


if __name__ == "__main__":
    main()
