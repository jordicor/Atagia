"""Cases and FROZEN card-1 ranges for the card 2 summary harness.

Two families:

- ``realistic``: the 5 conversational cases from
  ``benchmarks/model_casting/inputs/compactor.json``. Each carries a
  hand-authored, contiguous, non-overlapping partition of its seqs (these stand
  in for card-1 output, frozen so card 2 is measured in isolation).
- ``stress``: 2 synthetic stress cases with realistic fictional dialogue (no
  benchmark entities): one that saturates concurrency with ~20 single-message
  ranges, and one long single range (~30 messages) to confirm a big slice still
  summarizes.

The frozen ranges are versioned via ``FROZEN_RANGE_FIXTURE_ID`` so a report can
record exactly which partition produced its metrics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Bump this when the hand-authored partitions below change, so report
# artifacts can be matched against the exact fixture that produced them.
FROZEN_RANGE_FIXTURE_ID = "card2_frozen_ranges_v1"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMPACTOR_INPUTS = (
    _REPO_ROOT / "benchmarks" / "model_casting" / "inputs" / "compactor.json"
)


@dataclass(frozen=True, slots=True)
class Card2Case:
    """One case for the card 2 harness.

    ``messages`` are plain dicts with keys ``seq``, ``role``, ``occurred_at``,
    ``text`` -- the exact shape ``_summarize_message_ranges_card`` slices. The
    ``ranges`` are the frozen card-1 partition: contiguous, non-overlapping,
    covering every seq exactly once.
    """

    case_id: str
    family: str
    reference_time_utc: str
    messages: list[dict[str, Any]]
    ranges: list[tuple[int, int]]


# ---------------------------------------------------------------------------
# Realistic cases: frozen partitions (card-1 stand-in) for compactor.json.
# Authored from the message topics; every seq is covered exactly once.
# ---------------------------------------------------------------------------
_REALISTIC_RANGES: dict[str, list[tuple[int, int]]] = {
    # greeting -> work/weather -> walks/lake -> plan to meet
    "cmp_short_single_thread": [(1, 2), (3, 4), (5, 6), (7, 8)],
    # support group -> pottery/piano -> sister's baby
    "cmp_multi_topic_medium": [(1, 3), (4, 7), (8, 12)],
    # one deep thread about mentoring a teen
    "cmp_single_topic_deep": [(1, 6), (7, 10)],
    # plumber -> wedding -> journaling -> dog -> promotion -> recipe -> goodbye
    "cmp_rapid_topic_shifts": [(1, 2), (3, 4), (5, 6), (7, 7), (8, 10), (11, 12), (13, 15)],
    # venting about work -> coffee plan
    "cmp_monologue_and_reactions": [(1, 5), (6, 8)],
}


def _load_realistic_cases() -> list[Card2Case]:
    payload = json.loads(_COMPACTOR_INPUTS.read_text(encoding="utf-8"))
    cases: list[Card2Case] = []
    for raw in payload.get("cases", []):
        case_id = str(raw["case_id"])
        ranges = _REALISTIC_RANGES.get(case_id)
        if ranges is None:
            raise ValueError(
                f"No frozen range partition authored for realistic case {case_id!r}. "
                "Add one to _REALISTIC_RANGES (and bump FROZEN_RANGE_FIXTURE_ID)."
            )
        messages = [
            {
                "seq": int(message["seq"]),
                "role": str(message["role"]),
                "occurred_at": message.get("occurred_at"),
                "text": str(message["text"]),
            }
            for message in raw["messages"]
        ]
        cases.append(
            Card2Case(
                case_id=case_id,
                family="realistic",
                reference_time_utc=str(raw["reference_time_utc"]),
                messages=messages,
                ranges=list(ranges),
            )
        )
    return cases


def _stress_saturation_case() -> Card2Case:
    """~20 short single-message ranges to saturate the concurrency cap."""
    lines = [
        "I need to reorganize the sprint board before standup.",
        "Sure -- start by moving the blocked tickets to their own column.",
        "The staging deploy failed again last night.",
        "Check the migration step; it timed out on the new index.",
        "I booked the venue for the team offsite in March.",
        "Nice. Did you confirm the catering headcount?",
        "My laptop keyboard has a sticky spacebar now.",
        "Pop the keycap off and blow it out; that usually fixes it.",
        "We lost two subscribers after the price change.",
        "Worth sending a short win-back email before the weekend.",
        "I finally finished reading that systems book.",
        "Which one? The one on backpressure?",
        "The onboarding doc is out of date in three places.",
        "Flag them inline and I'll do a pass tomorrow.",
        "My flight on Friday got moved two hours earlier.",
        "Tight, but you'll still make the afternoon review.",
        "The analytics dashboard is showing duplicate events.",
        "Sounds like the SDK is firing twice on route change.",
        "Let's freeze the schema before the audit on Monday.",
        "Agreed -- I'll tag the release tonight.",
    ]
    messages = [
        {
            "seq": index + 1,
            "role": "user" if index % 2 == 0 else "assistant",
            "occurred_at": f"2026-05-01T{9 + index // 6:02d}:{(index * 7) % 60:02d}:00+00:00",
            "text": text,
        }
        for index, text in enumerate(lines)
    ]
    ranges = [(seq, seq) for seq in range(1, len(messages) + 1)]
    return Card2Case(
        case_id="stress_many_single_ranges",
        family="stress",
        reference_time_utc="2026-05-02T12:00:00+00:00",
        messages=messages,
        ranges=ranges,
    )


def _stress_long_single_range_case() -> Card2Case:
    """One long single range (~30 messages) so a big slice still summarizes."""
    topic_lines = [
        "Let's plan the data-warehouse migration end to end.",
        "Good -- first, what's the source of truth today?",
        "A single Postgres instance plus a few CSV exports.",
        "Then step one is cataloging every downstream consumer.",
        "I count seven dashboards and three nightly jobs.",
        "We should snapshot the schema before touching anything.",
        "Already done; the snapshot is in the shared drive.",
        "Next, stand up the warehouse in a sandbox project.",
        "Provisioned. It's empty but reachable from the VPC.",
        "Backfill historical data in batches by month.",
        "Batching by month keeps each load under the timeout.",
        "Validate row counts after each batch against the source.",
        "I'll write a reconciliation query for that check.",
        "Then dual-write for two weeks to catch drift.",
        "Dual-write is risky if the jobs aren't idempotent.",
        "Fair -- we'll add idempotency keys to the writers first.",
        "Cut the read traffic over one dashboard at a time.",
        "Start with the least critical internal dashboard.",
        "Monitor query latency for regressions during cutover.",
        "Set an alert if p95 latency doubles from baseline.",
        "After all reads move, freeze writes to the old DB.",
        "We'll keep it read-only for a month as a fallback.",
        "Document the rollback steps before the freeze.",
        "Rollback is: repoint the writers and replay the WAL.",
        "Schedule the final cutover for a low-traffic weekend.",
        "The last Sunday of the month works for everyone.",
        "Make sure on-call is staffed through Monday morning.",
        "I'll put two engineers on the rotation that weekend.",
        "Last thing: announce the maintenance window company-wide.",
        "I'll draft the announcement and send it for review.",
    ]
    messages = [
        {
            "seq": index + 1,
            "role": "user" if index % 2 == 0 else "assistant",
            "occurred_at": f"2026-05-03T14:{index:02d}:00+00:00",
            "text": text,
        }
        for index, text in enumerate(topic_lines)
    ]
    ranges = [(1, len(messages))]
    return Card2Case(
        case_id="stress_long_single_range",
        family="stress",
        reference_time_utc="2026-05-04T12:00:00+00:00",
        messages=messages,
        ranges=ranges,
    )


def stress_cases() -> list[Card2Case]:
    return [_stress_saturation_case(), _stress_long_single_range_case()]


def realistic_cases() -> list[Card2Case]:
    return _load_realistic_cases()


def load_cases(case_set: str) -> list[Card2Case]:
    """Return cases for ``all`` | ``realistic`` | ``stress``.

    Validates each case's frozen ranges cover every seq exactly once with no
    overlap, so a bad partition fails fast before any LLM call.
    """
    if case_set == "realistic":
        cases = realistic_cases()
    elif case_set == "stress":
        cases = stress_cases()
    elif case_set == "all":
        cases = [*realistic_cases(), *stress_cases()]
    else:
        raise ValueError(f"Unknown case set: {case_set!r} (use all | realistic | stress)")
    for case in cases:
        _validate_partition(case)
    return cases


def _validate_partition(case: Card2Case) -> None:
    message_seqs = sorted(int(message["seq"]) for message in case.messages)
    if not message_seqs:
        raise ValueError(f"Case {case.case_id!r} has no messages")
    covered: list[int] = []
    previous_end: int | None = None
    for start_seq, end_seq in case.ranges:
        if start_seq > end_seq:
            raise ValueError(
                f"Case {case.case_id!r} range {(start_seq, end_seq)} is inverted"
            )
        if previous_end is not None and start_seq <= previous_end:
            raise ValueError(
                f"Case {case.case_id!r} ranges overlap or are out of order at "
                f"{(start_seq, end_seq)}"
            )
        previous_end = end_seq
        covered.extend(range(start_seq, end_seq + 1))
    if covered != message_seqs:
        raise ValueError(
            f"Case {case.case_id!r} frozen ranges do not cover every seq exactly "
            f"once: covered={covered} message_seqs={message_seqs}"
        )
