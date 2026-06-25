"""Write JSON + markdown artifacts for a card 2 summary harness run."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from benchmarks.card2_summary.runner import RunReport


def write_report(report: RunReport, output_dir: Path) -> tuple[Path, Path]:
    """Write ``report.json`` and ``report.md`` under ``output_dir``.

    Returns the (json_path, markdown_path) pair.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "report.json"
    markdown_path = output_dir / "report.md"
    json_path.write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    markdown_path.write_text(format_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def format_markdown(report: RunReport) -> str:
    lines: list[str] = []
    lines.append("# Card 2 (summary) harness report")
    lines.append("")
    lines.append(f"- model: `{report.model}`")
    lines.append(f"- concurrency cap: {report.concurrency}")
    lines.append(f"- runs: {report.runs}")
    lines.append(f"- case set: {report.case_set}")
    lines.append(f"- card_examples_enabled: {report.card_examples_enabled}")
    lines.append(f"- frozen range fixture: `{report.frozen_range_fixture_id}`")
    lines.append(f"- selftest (offline fake provider): {report.selftest}")
    lines.append(f"- generated at (UTC): {report.generated_at_utc}")
    lines.append("")

    aggregate = report.aggregate
    if aggregate:
        lines.append("## Aggregate")
        lines.append("")
        lines.append(
            f"- overall range coverage: "
            f"{aggregate['overall_range_coverage'] * 100:.1f}% "
            f"({aggregate['total_summaries_returned']}/{aggregate['total_ranges']})"
        )
        lines.append(f"- total retries: {aggregate['total_retry_count']}")
        lines.append(f"- hard-failure cases: {aggregate['hard_failure_case_count']}")
        lines.append(
            f"- total LLM calls: {aggregate['total_llm_call_count']}"
        )
        lines.append(
            f"- observed max concurrency: {aggregate['observed_max_concurrency']} "
            f"(cap {report.concurrency})"
        )
        lines.append(
            f"- total wall clock: {aggregate['total_wall_clock_ms']:.0f} ms"
        )
        lines.append(
            f"- rate-limit / transient / request errors: "
            f"{aggregate['rate_limit_error_count']} / "
            f"{aggregate['transient_error_count']} / "
            f"{aggregate['request_error_count']}"
        )
        lines.append("")

    lines.append("## Per-case")
    lines.append("")
    lines.append(
        "| case | family | ranges | coverage | retries | hard_fail | calls | "
        "max_conc | rl/transient/req | wall_ms |"
    )
    lines.append(
        "| --- | --- | ---: | ---: | ---: | :---: | ---: | ---: | :---: | ---: |"
    )
    for metrics in report.per_case:
        lines.append(
            f"| {metrics.case_id} | {metrics.family} | {metrics.range_count} | "
            f"{metrics.range_coverage * 100:.0f}% | {metrics.retry_count} | "
            f"{'YES' if metrics.hard_failure else 'no'} | {metrics.llm_call_count} | "
            f"{metrics.observed_max_concurrency} | "
            f"{metrics.rate_limit_error_count}/{metrics.transient_error_count}/"
            f"{metrics.request_error_count} | {metrics.wall_clock_ms:.0f} |"
        )
    lines.append("")
    return "\n".join(lines)
