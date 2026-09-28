"""Runtime LLM budget and health guardrails.

TWO KINDS OF RUN, TWO KINDS OF VERDICT.

A *bounded* run ends: a bulk rebuild processes a finite backlog and stops. Its
totals are a budget ("this rebuild may spend 10000 calls"), and a verdict on it
is final, because a spent budget does not un-spend.

The *runtime* run is the process itself and never ends. Every absolute quantity
accumulated over it -- calls, failed calls, tokens, cost -- is crossed sooner or
later by a perfectly healthy process, so none of them is a health signal there;
they are budgets on a run that has no budget. That is why the process-wide
absolute caps default to ``None``.

What is left for the runtime run is health, and health is a statement about
RECENT behavior. Two properties follow, and this module enforces both:

* Health must be measured over a bounded window. A ratio over a run's whole
  life goes numb: on a process that has served a million calls, a live outage of
  the last hundred barely moves it, so the guard that is supposed to catch the
  outage sleeps through it. Ratios here are therefore computed over the last
  ``health_window_calls`` calls, and the consecutive-failure check reads the
  CURRENT streak rather than the worst streak ever seen (a high-water mark is
  monotonic, which is the same defect as an absolute cap wearing a health
  signal's name).
* A health verdict must expire, and expiring must not cost a fresh diagnosis.
  Blocking stops every call, which stops the very counters that would clear the
  verdict, so a guard that blocks forever after a bad minute is a latch, not a
  breaker: a provider having a bad hour on Monday wedges the process on Friday.
  A run configured with ``recovery_seconds`` therefore reopens -- but it reopens
  HALF-OPEN. One call per window is admitted as a PROBE while every other caller
  stays refused, and the probe's outcome decides: a success clears the health
  signals and reopens the run for everyone, a failure re-blocks for the next
  backoff step and KEEPS the evidence. Budget verdicts never expire and never
  probe -- and because a probe is the one call a blocked run spends, budgets are
  re-evaluated at probe admission, which is the only moment a blocked run can
  notice a budget crossed by the calls that were in flight when it closed.

  Admission is driven by a TIMESTAMP rather than by an "a probe is out" flag, so
  a ticket that is issued and never recorded cannot wedge the run. The cost of
  that is that the window can elapse while a probe is still in flight -- a
  provider round-trip may legally outlive it -- so the run can hold more than one
  probe ticket at once. Each ticket therefore carries the run's PROBE GENERATION
  at admission, and a verdict is applied only while that stamp is still current.
  A stale probe's spend and its health evidence are still recorded; what it
  cannot do is decide. Otherwise the probe that RETURNS last would win instead of
  the one admitted last, which reopens a run mid-outage on evidence a newer probe
  has already contradicted.

  Reopening fully instead -- clearing the window and every streak the moment the
  timer elapses -- makes every recovery cycle amnesiac: the run then needs a
  whole fresh ``min_calls_for_failed_ratio`` sample, or a whole fresh streak,
  before it can trip again, so a provider that stays down costs 9-20 failing
  calls per cycle, every ``recovery_seconds``, for as long as the outage lasts.
  Half-open probing costs ONE call per cycle, and the cycles get longer: the
  block doubles from ``recovery_seconds`` up to a ceiling of eight times it, so
  an hours-long outage is measured in tens of probes instead of thousands of
  failures.

WHAT A ``max_*`` THRESHOLD PROMISES.

Counts and ratios cannot promise the same thing, so each says what it means
instead of being read as one family:

* A ``max_*`` COUNT names the largest value the counter may reach. Reaching it
  closes the run, so the counter never goes past the name:
  ``max_total_calls=10000`` allows 10000 calls and refuses the 10001st, and
  ``max_consecutive_failures_per_purpose=8`` trips on the 8th consecutive
  failure rather than the 9th.
* A ``max_*`` RATIO names the largest ACCEPTABLE value. The ratio may sit on the
  threshold and the run closes on the first observation above it. A ratio cannot
  promise more: it moves by whatever the window does, so there is no "the next
  call would take it past" to refuse.

The wrinkle is WHEN a count closes the run. A budget is spend, and a run that
spends exactly its budget and then stops did nothing wrong, so budget counts are
enforced at the next ``begin_call`` (``>=``) and are not a violation at record
time (``>``): a bulk run that used all 10000 of its calls reports clean. A
health count is a diagnosis rather than spend, and its recovery clock has to
start when the storm is detected instead of whenever traffic next happens to
arrive, so the consecutive-failure count trips at record time the moment it is
reached.

That asymmetry is why EVERY path that spends a call has to evaluate budgets, not
just the ordinary one. A budget resting exactly on its limit is invisible at
record time by construction, so a path that admits a call without a pre-call
evaluation admits it past the budget -- which is what the half-open probe used
to do, once per recovery cycle, for as long as the outage lasted.

TWO INVARIANTS ABOUT *WHEN* AND *WHERE* A CALL IS CHARGED.

* THE GUARD DECIDES BEFORE SPENDING, NEVER AFTER. A violation can only be
  discovered by recording a call, and the call that reveals it is often a
  SUCCESS: a success is non-decreasing on a failure ratio, yet it is the sample
  that finally reaches ``min_calls_for_failed_ratio``, so the ratio is evaluated
  for the first time and trips. Turning that into an error would throw away an
  answer the provider already generated and the operator already paid for, and
  would do it to whichever user happened to arrive after someone else's outage.
  Recording therefore only LATCHES the verdict on the run; the block is applied
  by the next ``begin_call``. Everything the guard enforces is a statement about
  calls that already happened, so deferring the block by exactly one call costs
  one call and never loses a paid-for answer.
* A CALL IS CHARGED TO THE RUN IT WAS CHECKED AGAINST. ``begin_call`` returns an
  ``LLMRunGuardCall`` holding that run, and the outcome is recorded through it.
  Re-selecting the active run at completion time would let an operator reset (or
  a scope exit) that lands mid-flight move a straggler's failure onto a fresh
  run -- immediately re-tripping it on evidence from an outage it never served.
"""

from __future__ import annotations

from collections import Counter, deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter
from typing import Any, Iterator

# Backoff for repeated half-open probes. Each probe that fails multiplies the
# next block by this factor, so a provider that stays down is retried less and
# less often instead of at a fixed cadence. The ceiling exists because an
# unbounded backoff eventually turns a recoverable breaker back into the latch
# the recovery window was added to avoid: with a 60s base the block tops out at
# eight minutes, which is short enough that a recovered provider is noticed
# without an operator reset and long enough that a three-hour outage costs tens
# of probes rather than thousands of calls.
_RECOVERY_BACKOFF_FACTOR = 2.0
_RECOVERY_BACKOFF_CEILING_MULTIPLE = 8.0


class LLMCallOutcome(Enum):
    """How one provider round-trip ended.

    ``CANCELLED`` is neither of the other two: the round-trip happened and was
    paid for, but nobody consumed it (the caller was cancelled, or a proxy
    client disconnected mid-stream). It is SPEND, so it counts as a call. It is
    not EVIDENCE about provider health in either direction -- the provider was
    working, the client left -- so it stays out of the health window and out of
    the failure streak. Counting disconnects as failures would let a flaky
    network on the client side trip the breaker for everyone; counting them as
    successes would quietly certify a provider nobody actually heard back from.
    """

    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class LLMRunGuardConfig:
    """Thresholds that stop unhealthy or runaway LLM usage.

    The defaults describe the PROCESS-WIDE runtime run, which is what
    ``LLMRunGuard`` builds when no config is supplied: no absolute budget (a run
    that never ends cannot have one) and a recovery window so a failure storm
    blocks while it is happening instead of forever. Bounded runs are built
    explicitly by ``bulk_ingest_llm_run_guard_config``.
    """

    enabled: bool = True
    mode: str = "enforce"
    # Absolute budgets. Meaningful only on a run that ends; a violation of one
    # is FINAL and needs an operator reset.
    max_total_calls: int | None = None
    max_total_failed_calls: int | None = None
    max_failed_calls_per_purpose: int | None = None
    max_total_tokens: int | None = None
    max_reported_cost_usd: float | None = None
    max_wall_time_seconds: float | None = None
    # Health signals. Bounded, self-clearing, and recoverable.
    max_failed_call_ratio: float | None = 0.50
    min_calls_for_failed_ratio: int = 20
    max_failed_ratio_per_purpose: float | None = 0.50
    min_calls_per_purpose_for_failed_ratio: int = 10
    max_consecutive_failures_per_purpose: int | None = 8
    # How many of the most recent calls a failure ratio is computed over, per
    # run and per purpose. Bounds the ratio's memory so it keeps answering "is
    # it failing NOW" however long the process lives.
    health_window_calls: int = 200
    # How long a health verdict blocks before the run admits one probe call.
    # Every probe that fails multiplies the next block by
    # ``_RECOVERY_BACKOFF_FACTOR``, capped at
    # ``_RECOVERY_BACKOFF_CEILING_MULTIPLE`` times this value. ``None`` marks a
    # BOUNDED run: its verdict is final and it never probes.
    recovery_seconds: float | None = 60.0

    def __post_init__(self) -> None:
        if self.health_window_calls <= 0:
            raise ValueError("health_window_calls must be positive")
        if self.recovery_seconds is not None and self.recovery_seconds <= 0.0:
            raise ValueError("recovery_seconds must be positive when set")
        # A budget of N permits N, so a budget of 0 permits nothing and closes
        # the run before it serves a single call. That is never what an operator
        # writing "tolerate zero failures" means -- they mean "stop at the first
        # one", which is 1 -- and a value that can only produce a dead process
        # must not be accepted silently. ``Settings`` already rejects these;
        # this is the same rule for configs built in code.
        for name in (
            "max_total_calls",
            "max_total_failed_calls",
            "max_failed_calls_per_purpose",
            "max_consecutive_failures_per_purpose",
            "max_total_tokens",
            "max_reported_cost_usd",
            "max_wall_time_seconds",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(
                    f"{name} must be positive when set; use None to disable it"
                )
        if (
            self.max_failed_call_ratio is not None
            and self.min_calls_for_failed_ratio > self.health_window_calls
        ):
            raise ValueError(
                "min_calls_for_failed_ratio "
                f"({self.min_calls_for_failed_ratio}) exceeds health_window_calls "
                f"({self.health_window_calls}); the run failure ratio could never "
                "be evaluated"
            )
        if (
            self.max_failed_ratio_per_purpose is not None
            and self.min_calls_per_purpose_for_failed_ratio > self.health_window_calls
        ):
            raise ValueError(
                "min_calls_per_purpose_for_failed_ratio "
                f"({self.min_calls_per_purpose_for_failed_ratio}) exceeds "
                f"health_window_calls ({self.health_window_calls}); the per-purpose "
                "failure ratio could never be evaluated"
            )

    @classmethod
    def disabled(cls) -> "LLMRunGuardConfig":
        return cls(
            enabled=False,
            mode="off",
            max_total_calls=None,
            max_total_failed_calls=None,
            max_failed_call_ratio=None,
            min_calls_for_failed_ratio=0,
            max_failed_calls_per_purpose=None,
            max_failed_ratio_per_purpose=None,
            min_calls_per_purpose_for_failed_ratio=0,
            max_consecutive_failures_per_purpose=None,
            max_total_tokens=None,
            max_reported_cost_usd=None,
            max_wall_time_seconds=None,
            recovery_seconds=None,
        )

    def normalized_mode(self) -> str:
        mode = self.mode.strip().lower()
        return mode if mode in {"off", "audit", "enforce"} else "enforce"

    def is_enforcing(self) -> bool:
        return self.enabled and self.normalized_mode() == "enforce"


def runtime_llm_run_guard_config(settings: Any) -> LLMRunGuardConfig:
    """Build the default runtime LLM guard config from Settings-like objects."""
    return LLMRunGuardConfig(
        enabled=bool(settings.llm_run_guard_enabled),
        mode=str(settings.llm_run_guard_mode),
        max_total_calls=settings.llm_run_guard_max_total_calls,
        max_total_failed_calls=settings.llm_run_guard_max_total_failed_calls,
        max_failed_call_ratio=settings.llm_run_guard_max_failed_call_ratio,
        min_calls_for_failed_ratio=settings.llm_run_guard_failed_ratio_min_calls,
        max_failed_calls_per_purpose=settings.llm_run_guard_max_failed_calls_per_purpose,
        max_failed_ratio_per_purpose=settings.llm_run_guard_max_failed_ratio_per_purpose,
        min_calls_per_purpose_for_failed_ratio=(
            settings.llm_run_guard_purpose_failure_ratio_min_calls
        ),
        max_consecutive_failures_per_purpose=(
            settings.llm_run_guard_max_consecutive_failures_per_purpose
        ),
        max_total_tokens=settings.llm_run_guard_max_total_tokens,
        max_reported_cost_usd=settings.llm_run_guard_max_reported_cost_usd,
        max_wall_time_seconds=None,
        health_window_calls=settings.llm_run_guard_health_window_calls,
        recovery_seconds=settings.llm_run_guard_recovery_seconds,
    )


def bulk_ingest_llm_run_guard_config(settings: Any) -> LLMRunGuardConfig:
    """Build the stricter scoped LLM guard used for bulk/admin rebuilds."""
    return LLMRunGuardConfig(
        enabled=bool(settings.bulk_ingest_llm_run_guard_enabled),
        mode=str(settings.llm_run_guard_mode),
        max_total_calls=settings.bulk_ingest_llm_run_guard_max_total_calls,
        max_total_failed_calls=settings.bulk_ingest_llm_run_guard_max_total_failed_calls,
        max_failed_call_ratio=settings.bulk_ingest_llm_run_guard_max_failed_call_ratio,
        min_calls_for_failed_ratio=(
            settings.bulk_ingest_llm_run_guard_failed_ratio_min_calls
        ),
        max_failed_calls_per_purpose=(
            settings.bulk_ingest_llm_run_guard_max_failed_calls_per_purpose
        ),
        max_failed_ratio_per_purpose=(
            settings.bulk_ingest_llm_run_guard_max_failed_ratio_per_purpose
        ),
        min_calls_per_purpose_for_failed_ratio=(
            settings.bulk_ingest_llm_run_guard_purpose_failure_ratio_min_calls
        ),
        max_consecutive_failures_per_purpose=(
            settings.bulk_ingest_llm_run_guard_max_consecutive_failures_per_purpose
        ),
        max_total_tokens=settings.bulk_ingest_llm_run_guard_max_total_tokens,
        max_reported_cost_usd=settings.bulk_ingest_llm_run_guard_max_reported_cost_usd,
        max_wall_time_seconds=settings.bulk_ingest_llm_run_guard_max_wall_time_seconds,
        health_window_calls=settings.llm_run_guard_health_window_calls,
        # A bulk rebuild is BOUNDED: it has a budget and it ends, so its verdict
        # is final. Reopening it mid-run would let a rebuild that already blew
        # its budget quietly keep spending.
        recovery_seconds=None,
    )


@dataclass(frozen=True, slots=True)
class LLMRunGuardDecision:
    """Current guard decision plus a JSON-safe state snapshot.

    ``should_block`` is answered from the run's state and therefore means the
    same thing wherever it is read: THE RUN IS CLOSED TO CALLS. Only
    ``begin_call`` may act on it by refusing a call. On a decision returned by a
    ``record_*`` method it reports that the outcome just recorded closed the run,
    i.e. the NEXT call will be refused -- the call that produced this decision is
    already spent and its result belongs to its caller.

    ``tripped`` marks the ONE decision on which the run crossed from healthy to
    violating. It exists because ``should_block`` cannot carry that event in
    every mode: audit mode reports ``should_block=False`` by definition, so a
    log keyed on it never records the moment the guard would have fired -- which
    is the only thing audit mode exists to observe. Keyed on ``tripped``, the
    trip is logged exactly once per trip in both modes, instead of never in
    audit and once per subsequent decision if the log were keyed on
    ``violations`` being non-empty.
    """

    healthy: bool
    should_block: bool
    tripped: bool = False
    violations: tuple[str, ...] = ()
    snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _RecentOutcomes:
    """Bounded ring of the most recent call outcomes (``True`` == failed).

    Exists so a failure ratio keeps meaning something on a long-lived process.
    A lifetime ratio is monotonically desensitized by success: after a million
    healthy calls no outage short of a million failures can move it past a
    threshold, so the signal silently stops working on exactly the deployments
    it is supposed to protect.

    ``failed`` is maintained incrementally rather than recounted per decision:
    the ring is consulted on every provider round-trip.
    """

    capacity: int
    failed: int = field(init=False, default=0)
    _outcomes: deque[bool] = field(init=False)

    def __post_init__(self) -> None:
        self._outcomes = deque(maxlen=self.capacity)

    @property
    def calls(self) -> int:
        return len(self._outcomes)

    def record(self, *, failed: bool) -> None:
        if len(self._outcomes) == self.capacity and self._outcomes[0]:
            self.failed -= 1
        self._outcomes.append(failed)
        if failed:
            self.failed += 1

    def clear(self) -> None:
        self._outcomes.clear()
        self.failed = 0

    def failure_ratio(self) -> float:
        return self.failed / len(self._outcomes) if self._outcomes else 0.0


@dataclass(slots=True)
class _PurposeCounters:
    capacity: int
    calls: int = 0
    failed_calls: int = 0
    cancelled_calls: int = 0
    latency_ms: float = 0.0
    consecutive_failures: int = 0
    max_consecutive_failures: int = 0
    health: _RecentOutcomes = field(init=False)

    def __post_init__(self) -> None:
        self.health = _RecentOutcomes(capacity=self.capacity)


@dataclass(slots=True)
class LLMRunGuardRun:
    """Mutable counters for one runtime or scoped LLM run.

    The plain totals are ACCOUNTING: they answer "what did this run spend" for
    admin and benchmark surfaces and are never cleared while the run lives.
    ``health`` and the per-purpose streaks are the HEALTH signals the guard
    actually enforces on, and those are cleared whenever the run recovers.
    Keeping the two separate is what lets a run reopen without erasing its
    ledger.
    """

    run_id: str
    kind: str
    config: LLMRunGuardConfig
    started_at_monotonic: float
    status: str = "active"
    total_calls: int = 0
    failed_calls: int = 0
    # Round-trips abandoned by their caller. Included in ``total_calls`` because
    # they were paid for, excluded from ``failed_calls`` and from every health
    # signal because they say nothing about the provider.
    cancelled_calls: int = 0
    total_tokens: float = 0.0
    reported_cost_usd: float = 0.0
    # Round-trips that SUCCEEDED and still reported no tokens / no cost. Without
    # these a token or cost budget that can never fire -- because the provider
    # omits the usage the budget is measured in -- is indistinguishable from a
    # budget that is simply not close to its ceiling. Only successes count: a
    # failed or cancelled round-trip has no usage to report by construction, so
    # counting those would make the number non-zero on every run and useless.
    successful_calls_without_token_usage: int = 0
    successful_calls_without_reported_cost: int = 0
    total_latency_ms: float = 0.0
    model_counts: Counter[str] = field(default_factory=Counter)
    error_counts: Counter[str] = field(default_factory=Counter)
    purposes: dict[str, _PurposeCounters] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)
    # Set when a BUDGET was blown, which is a statement about spend and does not
    # expire. A health verdict leaves this False and carries an expiry instead.
    verdict_is_final: bool = False
    blocked_until_monotonic: float | None = None
    recovery_count: int = 0
    # Half-open probing. ``health_block_streak`` counts the health blocks this
    # run has served without a probe clearing them, and is what the recovery
    # backoff is computed from; a successful probe resets it to zero.
    # ``probe_in_flight`` is observability only -- enforcement is driven by
    # ``blocked_until_monotonic``, which is re-armed when a probe is admitted so
    # that a probe whose ticket is never recorded cannot wedge the run.
    #
    # ``probe_generation`` is what makes a probe's VERDICT single-valued while
    # its ADMISSION stays timer-driven. The re-armed window can elapse with a
    # probe still in flight -- a provider round-trip may legally take twice the
    # window (120s request timeout against a 60s default recovery), and a
    # streamed proxy reply holds its ticket for the whole stream -- so two
    # probes can overlap. Each admission stamps the ticket with the value here;
    # a resolution whose stamp no longer matches decides nothing. Without it the
    # LATER-RETURNING probe wins regardless of which was admitted last: a stale
    # success reopens a run that a newer failure just re-blocked, and a stale
    # failure arms a recovery window on a run that is already healthy again --
    # which turns the next ordinary call into a phantom probe that skips its own
    # pre-call decision and, on success, wipes the health ring of a healthy run.
    health_block_streak: int = 0
    probe_in_flight: bool = False
    probe_generation: int = 0
    # Whether the current violating state has already been reported. Turns a
    # standing violation into a single trip EVENT, which is what a log wants.
    unhealthy_reported: bool = False
    health: _RecentOutcomes = field(init=False)

    def __post_init__(self) -> None:
        self.health = _RecentOutcomes(capacity=self.config.health_window_calls)

    def elapsed_seconds(self, now: float) -> float:
        return max(0.0, now - self.started_at_monotonic)


_CURRENT_RUN: ContextVar[LLMRunGuardRun | None] = ContextVar(
    "atagia_llm_run_guard_current_run",
    default=None,
)


@dataclass(frozen=True, slots=True)
class LLMRunGuardCall:
    """One checked, not-yet-recorded provider round-trip.

    Binds the pre-call check to the post-call record. Holding the run (rather
    than looking the active one up again when the provider returns) is what
    keeps an outcome on the run that authorized it: ``reset_runtime`` replaces
    the process-wide run wholesale, and a scoped run ends when its context
    manager exits, so "the active run" at completion time is not necessarily the
    run the call was measured against.
    """

    run: LLMRunGuardRun
    purpose: str | None
    request_model: str
    decision: LLMRunGuardDecision
    # The run's probe generation at admission, or ``None`` for an ordinary call.
    # Set for the single call a blocked run admits when its recovery window
    # elapses; that call's outcome, and only that call's outcome, decides
    # whether the run reopens. It travels with the ticket rather than being read
    # off the run at completion time for the same reason ``run`` does -- the run
    # may have been replaced or re-blocked while this call was in flight -- and
    # it is a generation rather than a flag so that a probe returning after the
    # run has moved on can be recognized as stale and decide nothing.
    probe_generation: int | None = None

    @property
    def is_probe(self) -> bool:
        return self.probe_generation is not None


class LLMRunGuard:
    """In-process guard for LLM call budgets and failure storms."""

    def __init__(
        self,
        default_config: LLMRunGuardConfig | None = None,
        *,
        now: Any = perf_counter,
    ) -> None:
        self._now = now
        self._default_run = LLMRunGuardRun(
            run_id="runtime",
            kind="runtime",
            config=default_config or LLMRunGuardConfig(),
            started_at_monotonic=float(self._now()),
        )
        self._last_scoped_snapshot: dict[str, Any] | None = None

    @contextmanager
    def scoped_run(
        self,
        *,
        run_id: str,
        kind: str,
        config: LLMRunGuardConfig | None = None,
    ) -> Iterator[LLMRunGuardRun]:
        """Apply a separate budget to LLM calls made inside the context.

        SCOPE LEAKS INTO TASKS SPAWNED INSIDE IT. ``_CURRENT_RUN.set`` is
        context-local and ``asyncio.create_task`` copies the context, so a task
        spawned in here keeps this run after the scope exits -- and a bulk
        config has ``recovery_seconds=None``, which makes its verdict permanent,
        so a child task that outlives the scope would charge a dead run and, once
        that run is blocked, raise on every later call forever. No production
        ``scoped_run`` currently spawns LLM-calling work (checked:
        ``admin_rebuild_service`` awaits its work inline). The first one that
        does must bind its own run in the child (or await the child inside the
        scope); this comment is the warning to whoever writes it.
        """
        run = LLMRunGuardRun(
            run_id=run_id,
            kind=kind,
            config=config or self._default_run.config,
            started_at_monotonic=float(self._now()),
        )
        token = _CURRENT_RUN.set(run)
        try:
            yield run
        finally:
            self._last_scoped_snapshot = self.snapshot(run)
            _CURRENT_RUN.reset(token)

    def maybe_scoped_run(
        self,
        *,
        run_id: str,
        kind: str,
        config: LLMRunGuardConfig | None = None,
    ) -> Any:
        return self.scoped_run(run_id=run_id, kind=kind, config=config)

    def reset_runtime(self) -> dict[str, Any]:
        """Reset the process-wide guard after an operator resolves the cause."""
        self._default_run = LLMRunGuardRun(
            run_id="runtime",
            kind="runtime",
            config=self._default_run.config,
            started_at_monotonic=float(self._now()),
        )
        return self.snapshot()

    def begin_call(
        self,
        *,
        purpose: str | None,
        request_model: str,
    ) -> LLMRunGuardCall:
        """Check the active run and return the ticket the outcome is recorded on.

        This is the ONLY place a call is refused. The returned decision must be
        acted on before the provider is touched, and the returned ticket must be
        handed back to exactly one ``record_*`` call.

        CHECK-THEN-ACT IS NOT ATOMIC, BY DESIGN. Concurrent callers all read the
        same counters here and all pass, so a run can overshoot its thresholds by
        up to the real fan-out width of one turn (8 need-detection cards, 6
        extraction, 5 consequence, 4 applicability-card batches). That is
        accepted for the health signals: a windowed ratio is approximate by
        contract, and a single-digit overshoot on a 200-call window is noise the
        design already absorbs.

        AN EXACT CALL BUDGET IS AVAILABLE WITHOUT LOCKS, AND IS DELIBERATELY NOT
        BUILT. This method is synchronous -- there is no ``await`` between
        reading the counters and returning the ticket -- so within one event loop
        it already runs to completion atomically. An ``in_flight`` counter
        incremented here and decremented in ``_record_call``, with the check
        reading ``total_calls + in_flight``, would drop the call-budget overshoot
        from the fan-out width to zero with nothing held across I/O. It is not
        built because there is nothing to protect yet: ``max_total_calls``
        defaults to ``None`` on the runtime run and bulk rebuilds are
        near-sequential. Build it the day an exact call budget under concurrency
        is a real requirement -- the alternative is NOT "a lock across a provider
        round-trip", which is what the shape of this comment used to imply.
        """
        run = self._active_run()
        probe_generation: int | None = None
        if self._probe_is_due(run):
            # A probe asks a HEALTH question, so it must not be the hole a spent
            # BUDGET walks through. Probe admission is also the only moment a
            # blocked run spends a call, which makes it the only moment a budget
            # crossed while the run was blocked can be seen at all.
            decision = self._spent_budget_decision(run)
            if decision is None:
                probe_generation = self._admit_half_open_probe(run)
                # Not healthy -- the violations that blocked the run still stand
                # -- but not blocked either: this one call is how the run finds
                # out whether they still describe reality.
                decision = LLMRunGuardDecision(
                    healthy=False,
                    should_block=False,
                    violations=tuple(run.violations),
                    snapshot=self._snapshot(run),
                )
        else:
            decision = self._decision(run, before_next_call=True)
        return LLMRunGuardCall(
            run=run,
            purpose=purpose,
            request_model=request_model,
            decision=decision,
            probe_generation=probe_generation,
        )

    def record_success(
        self,
        call: LLMRunGuardCall,
        *,
        usage: dict[str, Any] | None,
        latency_ms: float,
    ) -> LLMRunGuardDecision:
        """Record a completed round-trip and re-evaluate its run.

        The returned decision may report the run as blocked; that governs the
        NEXT call, never this one. See ``LLMRunGuardDecision``.
        """
        return self._record_outcome(
            call,
            usage=usage or {},
            latency_ms=latency_ms,
            outcome=LLMCallOutcome.SUCCESS,
            error_type=None,
        )

    def record_failure(
        self,
        call: LLMRunGuardCall,
        *,
        latency_ms: float,
        error_type: str,
    ) -> LLMRunGuardDecision:
        """Record a failed round-trip and re-evaluate its run."""
        return self._record_outcome(
            call,
            usage={},
            latency_ms=latency_ms,
            outcome=LLMCallOutcome.FAILURE,
            error_type=error_type or "UnknownError",
        )

    def record_cancellation(
        self,
        call: LLMRunGuardCall,
        *,
        usage: dict[str, Any] | None,
        latency_ms: float,
    ) -> LLMRunGuardDecision:
        """Record a round-trip abandoned by its caller and re-evaluate its run.

        Counted as spend, never as a health signal: see ``LLMCallOutcome``.

        CANCELLED SPEND IS UNDER-MEASURED, AND THAT IS A MEASUREMENT LIMIT
        RATHER THAN A CHOICE. The provider generated tokens and the operator is
        billed for them, but a cancelled stream never delivers the terminal
        chunk that carries the usage, so the numbers mostly do not exist at this
        point. ``usage`` is whatever the stream had already published before the
        caller went away: today that is empty for every provider adapter in this
        repo, because all of them report usage only on their terminal event even
        when the wire format carries a running total. The consequence is worth
        stating plainly -- a token or cost budget cannot see cancelled spend, so
        a disconnect storm spends money that no budget counts. The call itself
        is always counted, which is why ``cancelled_calls`` is on the snapshot.
        """
        return self._record_outcome(
            call,
            usage=usage or {},
            latency_ms=latency_ms,
            outcome=LLMCallOutcome.CANCELLED,
            error_type=None,
        )

    def snapshot(self, run: LLMRunGuardRun | None = None) -> dict[str, Any]:
        active_run = run or self._active_run()
        return self._snapshot(active_run)

    def runtime_snapshot(self) -> dict[str, Any]:
        snapshot = self._snapshot(self._default_run)
        if self._last_scoped_snapshot is not None:
            snapshot["last_scoped_run"] = dict(self._last_scoped_snapshot)
        return snapshot

    def _active_run(self) -> LLMRunGuardRun:
        return _CURRENT_RUN.get() or self._default_run

    def _record_outcome(
        self,
        call: LLMRunGuardCall,
        *,
        usage: dict[str, Any],
        latency_ms: float,
        outcome: LLMCallOutcome,
        error_type: str | None,
    ) -> LLMRunGuardDecision:
        # EVIDENCE IS RECORDED IN ARRIVAL ORDER, NOT IN CALL ORDER, AND THAT IS
        # A CHOSEN TRADE-OFF RATHER THAN AN OVERSIGHT. Round-trips overlap, so a
        # call that started earlier can finish later and land in the health ring
        # behind a newer one; the ring is then chronologically inverted at its
        # tail. Ordering by start time instead would mean buffering outcomes
        # until every earlier call resolved -- unbounded, and delaying the very
        # diagnosis the ring exists to make. Recording on arrival keeps the ring
        # a truthful count of what the provider did within the window, and only
        # its internal order approximate.
        #
        # What that order can distort is bounded to the tail: the failed RATIO
        # is order-independent, and only the consecutive-failure streak reads
        # sequence at all. With the shipped defaults (200-call window, 20-call
        # minimum, 8-failure streak) one out-of-order outcome moves neither. The
        # probe VERDICT is a separate question and is not left to arrival order
        # -- ``_resolve_half_open_probe`` fences it on the generation stamp, so
        # the probe admitted last wins rather than the one that returns last.
        # See ``test_a_stale_probe_is_still_charged_and_still_counted_as_evidence``.
        self._record_call(
            call.run,
            purpose=call.purpose,
            request_model=call.request_model,
            usage=usage,
            latency_ms=latency_ms,
            outcome=outcome,
            error_type=error_type,
        )
        if call.probe_generation is not None:
            self._resolve_half_open_probe(
                call.run,
                outcome=outcome,
                probe_generation=call.probe_generation,
            )
        return self._decision(call.run, before_next_call=False)

    def _record_call(
        self,
        run: LLMRunGuardRun,
        *,
        purpose: str | None,
        request_model: str,
        usage: dict[str, Any],
        latency_ms: float,
        outcome: LLMCallOutcome,
        error_type: str | None,
    ) -> None:
        """Fold one round-trip into the run's counters.

        EVERY MUTATION BELOW IS UNLOCKED, AND THAT IS CORRECT ONLY BECAUSE ALL
        CALLERS RUN ON ONE EVENT LOOP. ``+=`` is a read-modify-write, so the
        first sync client path that drives this guard from a thread pool starts
        losing updates on ``total_tokens``, ``total_calls`` and the rings --
        silently, as under-counting. If that path is ever added, this method (and
        ``begin_call``) need a lock, and the lock must not be held across the
        provider round-trip.
        """
        # A run keeps one counter set per DISTINCT purpose label for its whole
        # life, and the runtime run never ends. That is bounded today because
        # purposes are a closed vocabulary of stage names set by the engine; the
        # day a label carries anything caller-derived (a model name, a tenant, a
        # request id) this dict becomes an unbounded leak on a process-wide run.
        # Keep purpose labels a fixed vocabulary, or bound this map here.
        purpose_label = (purpose or "unknown").strip() or "unknown"
        counters = run.purposes.get(purpose_label)
        if counters is None:
            counters = _PurposeCounters(capacity=run.config.health_window_calls)
            run.purposes[purpose_label] = counters
        normalized_latency_ms = max(0.0, float(latency_ms))
        # Accounting first: every outcome, cancellations included, is spend.
        run.total_calls += 1
        counters.calls += 1
        counters.latency_ms += normalized_latency_ms
        run.total_latency_ms += normalized_latency_ms
        if request_model:
            run.model_counts[request_model] += 1
        call_tokens = _total_tokens(usage)
        call_cost = _reported_cost(usage)
        run.total_tokens += call_tokens
        run.reported_cost_usd += call_cost
        if outcome is LLMCallOutcome.SUCCESS:
            # A success with no reported tokens or no reported cost means the
            # provider did not publish the quantity, not that the call was free.
            if call_tokens <= 0.0:
                run.successful_calls_without_token_usage += 1
            if call_cost <= 0.0:
                run.successful_calls_without_reported_cost += 1
        if outcome is LLMCallOutcome.CANCELLED:
            # No health evidence in either direction, so the window and the
            # streak are left exactly as the last real outcome left them.
            run.cancelled_calls += 1
            counters.cancelled_calls += 1
            return
        failed = outcome is LLMCallOutcome.FAILURE
        run.health.record(failed=failed)
        counters.health.record(failed=failed)
        if not failed:
            counters.consecutive_failures = 0
            return
        run.failed_calls += 1
        run.error_counts[error_type or "UnknownError"] += 1
        counters.failed_calls += 1
        counters.consecutive_failures += 1
        counters.max_consecutive_failures = max(
            counters.max_consecutive_failures,
            counters.consecutive_failures,
        )

    def _decision(
        self,
        run: LLMRunGuardRun,
        *,
        before_next_call: bool,
    ) -> LLMRunGuardDecision:
        config = run.config
        if not config.enabled or config.normalized_mode() == "off":
            return LLMRunGuardDecision(
                healthy=True,
                should_block=False,
                snapshot=self._snapshot(run),
            )
        enforcing = config.is_enforcing()

        if run.status == "failed" and run.violations:
            # Still blocked: a final budget verdict, a health verdict whose
            # recovery window has not elapsed, or a health verdict whose probe is
            # out. A blocked run leaves this state ONLY through a probe that
            # succeeds, which is why nothing here re-evaluates or clears.
            #
            # NOT because the counters cannot have moved -- they can, and
            # ``_spent_budget_decision`` exists precisely because they do: calls
            # already in flight when the block landed are still recorded
            # afterwards, and every probe is itself spend. The reason is that
            # this is the wrong place to notice it. Re-evaluating here would
            # re-decide a standing verdict on every refused caller, at no benefit
            # to any of them; the one moment a moved counter changes anything is
            # when the run is about to SPEND again, which is the probe admission,
            # and that path evaluates budgets itself.
            return LLMRunGuardDecision(
                healthy=False,
                should_block=enforcing,
                violations=tuple(run.violations),
                snapshot=self._snapshot(run),
            )

        budget_violations, health_violations = self._evaluate(
            run,
            before_next_call=before_next_call,
        )
        violations = budget_violations + health_violations
        if not violations:
            run.status = "active"
            run.violations = []
            run.unhealthy_reported = False
            return LLMRunGuardDecision(
                healthy=True,
                should_block=False,
                snapshot=self._snapshot(run),
            )

        run.violations = violations
        tripped = not run.unhealthy_reported
        run.unhealthy_reported = True
        if not enforcing:
            # Audit mode never blocks, so it never starves its own counters and
            # needs no expiry: the next decision re-derives the verdict from
            # live signals and clears "degraded" on its own once health returns.
            run.status = "degraded"
            return LLMRunGuardDecision(
                healthy=False,
                should_block=False,
                tripped=tripped,
                violations=tuple(violations),
                snapshot=self._snapshot(run),
            )

        run.status = "failed"
        run.verdict_is_final = bool(budget_violations) or config.recovery_seconds is None
        if run.verdict_is_final:
            run.blocked_until_monotonic = None
        else:
            run.health_block_streak += 1
            run.blocked_until_monotonic = float(self._now()) + self._recovery_delay(run)
        return LLMRunGuardDecision(
            healthy=False,
            should_block=True,
            tripped=tripped,
            violations=tuple(violations),
            snapshot=self._snapshot(run),
        )

    def _recovery_delay(self, run: LLMRunGuardRun) -> float:
        """Return how long the CURRENT health block lasts before the next probe.

        Doubles per block already served, capped so the breaker stays a breaker.
        """
        base = float(run.config.recovery_seconds or 0.0)
        steps = max(0, run.health_block_streak - 1)
        return base * min(
            _RECOVERY_BACKOFF_FACTOR**steps,
            _RECOVERY_BACKOFF_CEILING_MULTIPLE,
        )

    def _probe_is_due(self, run: LLMRunGuardRun) -> bool:
        """Whether the run's recovery window has elapsed, without touching it.

        A window is only ever armed by an ENFORCING run: audit mode returns from
        ``_decision`` before the block deadline is set, and a final budget
        verdict nulls it. So a due probe also means "this run is enforcing and
        its verdict is a health verdict", which is what lets ``begin_call``
        evaluate budgets here without a second mode branch.
        """
        blocked_until = run.blocked_until_monotonic
        return blocked_until is not None and float(self._now()) >= blocked_until

    def _spent_budget_decision(
        self,
        run: LLMRunGuardRun,
    ) -> LLMRunGuardDecision | None:
        """Latch any budget the run has crossed since it was blocked.

        Returns the refusing decision, or ``None`` when every budget still has
        room and the probe may go ahead.

        THE STANDING BLOCK IS NOT A REASON TO STOP COUNTING. ``_decision``
        short-circuits on a standing block, on the reasoning that blocked
        counters cannot move -- but they move twice over. Calls already in
        flight when the block landed are recorded afterwards (the guard blocks
        BEFORE a call, never after one, so their answers are kept and charged),
        and every probe is itself spend. Budget counts are also latched at
        record time with ``>``, because a run that stopped exactly on its
        allowance did nothing wrong, so a budget sitting ON its limit is
        invisible until a pre-call evaluation runs -- and the probe path is
        precisely where no pre-call evaluation happened. The two together let a
        blocked run keep spending one call per recovery cycle for the whole
        outage, past a ``max_*`` count whose own promise is that the counter
        never goes past its name.

        A budget verdict is FINAL and never probes, so latching it here also
        ends the probing: the deadline is nulled and any ticket still in flight
        is invalidated, or a probe admitted before this budget was crossed could
        come back a minute later and reopen a run that has no budget left.
        """
        budget_violations, health_violations = self._evaluate(
            run,
            before_next_call=True,
        )
        if not budget_violations:
            return None
        violations = budget_violations + health_violations
        # Read the same way ``_decision`` reads it, so the two latch paths cannot
        # drift. A run with a due probe has already reported its health trip, so
        # this is False in practice: a budget closing an already-blocked run is a
        # harder verdict on the same incident, not a second trip EVENT.
        tripped = not run.unhealthy_reported
        run.unhealthy_reported = True
        run.violations = violations
        run.status = "failed"
        run.verdict_is_final = True
        run.blocked_until_monotonic = None
        run.probe_in_flight = False
        run.probe_generation += 1
        return LLMRunGuardDecision(
            healthy=False,
            should_block=True,
            tripped=tripped,
            violations=tuple(violations),
            snapshot=self._snapshot(run),
        )

    def _admit_half_open_probe(self, run: LLMRunGuardRun) -> int:
        """Admit the one call a due health block lets through, and re-arm.

        The caller has already established that the window elapsed
        (``_probe_is_due``) and that no budget refuses the call. Returns the
        generation stamped on the admitted probe's ticket.

        THE ALTERNATIVE IS AMNESIA. Reopening fully here -- clearing the window
        and every streak because "blocking starves the counters" -- is true only
        of a full reopen, and it makes the run rediscover a known outage from
        scratch on every cycle, at 9-20 failing calls each. Admitting ONE call
        instead keeps the evidence and costs one call to learn whether it is
        stale.

        ONLY ONE CALL PASSES PER WINDOW, and the timer is what enforces it: the
        window is re-armed here, before returning, so every other caller that
        arrives while the probe is out reads a block that has not elapsed. This
        method is synchronous, so within one event loop that re-arming is atomic
        with the check that precedes it. Re-arming is also why a probe whose
        ticket is never recorded cannot wedge the run: the state that gates the
        next probe is a timestamp that always elapses, never a flag that
        something has to clear.

        That property is about ADMISSION, and it is why a window can elapse with
        a probe still out -- so the run can hold more than one probe ticket at a
        time. The generation stamped here is what keeps the VERDICT single: see
        ``_resolve_half_open_probe``.
        """
        run.probe_generation += 1
        run.blocked_until_monotonic = float(self._now()) + self._recovery_delay(run)
        run.probe_in_flight = True
        return run.probe_generation

    def _resolve_half_open_probe(
        self,
        run: LLMRunGuardRun,
        *,
        outcome: LLMCallOutcome,
        probe_generation: int,
    ) -> None:
        """Apply the probe's verdict: reopen, back off, or neither.

        A STALE PROBE DECIDES NOTHING. Its spend is already recorded and its
        outcome is already in the health window -- it is real evidence about the
        provider -- but the run has since admitted a newer probe, or closed on a
        budget, so this ticket no longer speaks for the run's current state.
        Applying it anyway would let the LATER-RETURNING probe win rather than
        the later-ADMITTED one, which is how a 90-second success erases a newer
        failure and reopens general traffic in the middle of an outage.
        """
        if probe_generation != run.probe_generation:
            return
        run.probe_in_flight = False
        if outcome is LLMCallOutcome.CANCELLED:
            # A probe whose caller went away proves nothing in either direction,
            # exactly as a cancellation proves nothing about provider health --
            # so it neither reopens the run nor escalates the backoff. The window
            # re-armed at admission stands, and the first call after it elapses
            # becomes the next probe.
            return
        if outcome is LLMCallOutcome.FAILURE:
            run.health_block_streak += 1
            run.blocked_until_monotonic = float(self._now()) + self._recovery_delay(run)
            return
        self._reopen(run)

    def _reopen(self, run: LLMRunGuardRun) -> None:
        """Close the breaker after a probe proves the run can serve traffic.

        Clearing the health signals (and only those) is what gives the reopened
        run a real chance: leaving them would re-trip on the same evidence the
        probe just contradicted, and clearing the totals as well would erase the
        run's ledger. One success is enough -- requiring N would put the run back
        in the business of paying for a diagnosis it can get for one call.
        """
        run.blocked_until_monotonic = None
        run.status = "active"
        run.violations = []
        run.health_block_streak = 0
        run.recovery_count += 1
        run.health.clear()
        for counters in run.purposes.values():
            counters.consecutive_failures = 0
            counters.health.clear()

    def _evaluate(
        self,
        run: LLMRunGuardRun,
        *,
        before_next_call: bool,
    ) -> tuple[list[str], list[str]]:
        """Return ``(budget_violations, health_violations)`` for the run.

        Budget violations are absolute quantities accumulated over the run's
        whole life; they are final. Health violations describe the last
        ``health_window_calls`` calls and the current failure streak; they
        expire. The split is what lets one guard serve both a bulk job with a
        real budget and a process that has none.

        Every absolute budget goes through ``_budget_is_spent`` so that "max"
        means one thing across the whole config block. See the module docstring
        for what a ``max_*`` count promises and why a ``max_*`` ratio cannot
        promise the same.
        """
        config = run.config
        budget_violations: list[str] = []
        health_violations: list[str] = []
        total_calls = run.total_calls
        failed_calls = run.failed_calls

        if config.max_total_calls is not None and self._budget_is_spent(
            total_calls,
            config.max_total_calls,
            before_next_call=before_next_call,
        ):
            budget_violations.append(
                f"total LLM calls budget {config.max_total_calls} is spent: "
                f"{total_calls}"
            )

        if config.max_total_failed_calls is not None and self._budget_is_spent(
            failed_calls,
            config.max_total_failed_calls,
            before_next_call=before_next_call,
        ):
            budget_violations.append(
                "total failed LLM calls budget "
                f"{config.max_total_failed_calls} is spent: {failed_calls}"
            )

        if config.max_total_tokens is not None and self._budget_is_spent(
            run.total_tokens,
            config.max_total_tokens,
            before_next_call=before_next_call,
        ):
            budget_violations.append(
                f"total LLM tokens budget {config.max_total_tokens} is spent: "
                f"{int(run.total_tokens)}"
            )

        if config.max_reported_cost_usd is not None and self._budget_is_spent(
            run.reported_cost_usd,
            config.max_reported_cost_usd,
            before_next_call=before_next_call,
        ):
            budget_violations.append(
                "reported LLM cost budget "
                f"{config.max_reported_cost_usd:.4f} is spent: "
                f"{run.reported_cost_usd:.4f}"
            )

        if config.max_wall_time_seconds is not None:
            elapsed_seconds = run.elapsed_seconds(float(self._now()))
            if self._budget_is_spent(
                elapsed_seconds,
                config.max_wall_time_seconds,
                before_next_call=before_next_call,
            ):
                budget_violations.append(
                    "LLM run wall time budget "
                    f"{config.max_wall_time_seconds:.1f}s is spent: "
                    f"{elapsed_seconds:.1f}s"
                )

        recent = run.health
        if (
            config.max_failed_call_ratio is not None
            and recent.calls >= config.min_calls_for_failed_ratio
            and recent.calls > 0
            and recent.failure_ratio() > config.max_failed_call_ratio
        ):
            health_violations.append(
                "LLM failure ratio over the last "
                f"{recent.calls} calls exceeded "
                f"{config.max_failed_call_ratio:.2%}: "
                f"{recent.failed}/{recent.calls} ({recent.failure_ratio():.2%})"
            )

        for purpose, counters in sorted(run.purposes.items()):
            if config.max_failed_calls_per_purpose is not None and self._budget_is_spent(
                counters.failed_calls,
                config.max_failed_calls_per_purpose,
                before_next_call=before_next_call,
            ):
                budget_violations.append(
                    "failed LLM calls budget "
                    f"{config.max_failed_calls_per_purpose} for purpose "
                    f"{purpose!r} is spent: {counters.failed_calls}"
                )
            purpose_recent = counters.health
            if (
                config.max_failed_ratio_per_purpose is not None
                and purpose_recent.calls
                >= config.min_calls_per_purpose_for_failed_ratio
                and purpose_recent.calls > 0
                and purpose_recent.failure_ratio()
                > config.max_failed_ratio_per_purpose
            ):
                health_violations.append(
                    "LLM failure ratio for purpose "
                    f"{purpose!r} over the last {purpose_recent.calls} calls "
                    f"exceeded {config.max_failed_ratio_per_purpose:.2%}: "
                    f"{purpose_recent.failed}/{purpose_recent.calls} "
                    f"({purpose_recent.failure_ratio():.2%})"
                )
            # The CURRENT streak, not the worst streak ever seen. A high-water
            # mark only ever grows, so reading it here would mean one transient
            # burst of failures condemns the run for as long as it lives -- the
            # exact defect the absolute caps have, wearing a health signal's
            # name. The high-water mark stays in the snapshot as diagnostics.
            #
            # ``>=`` at BOTH moments, unlike the budgets above. A streak is a
            # diagnosis rather than spend, so there is no "the run stopped
            # exactly at its allowance and did nothing wrong" case to protect --
            # and deferring it to the next ``begin_call`` would start the
            # recovery clock whenever traffic next happened to arrive, so a
            # process idle for an hour after a storm would be blocked on its
            # first new call instead of having recovered while it was idle.
            if (
                config.max_consecutive_failures_per_purpose is not None
                and counters.consecutive_failures
                >= config.max_consecutive_failures_per_purpose
            ):
                health_violations.append(
                    "consecutive failed LLM calls for purpose "
                    f"{purpose!r} reached "
                    f"{config.max_consecutive_failures_per_purpose}: "
                    f"{counters.consecutive_failures}"
                )

        return budget_violations, health_violations

    @staticmethod
    def _budget_is_spent(
        value: float,
        limit: float,
        *,
        before_next_call: bool,
    ) -> bool:
        """Whether an absolute budget of ``limit`` closes the run at ``value``.

        A budget of N permits N. Reaching it refuses the NEXT call (``>=``)
        without making a run that stopped exactly there a violation (``>``): a
        bulk rebuild that used all 10000 of its calls and finished did what it
        was allowed to do, and must not report itself failed for it.

        Both forms are the same promise read at two moments, which is why every
        absolute budget goes through this one function. The previous shape
        applied ``>=`` to ``max_total_calls`` alone and left every other budget
        one unit looser than its own name -- ``max_total_failed_calls=1``
        tolerated two failures, ``max_failed_calls_per_purpose=10`` tolerated
        eleven -- so "max" meant two different things inside one settings block.
        """
        return value >= limit if before_next_call else value > limit

    def _snapshot(self, run: LLMRunGuardRun) -> dict[str, Any]:
        config = run.config
        now = float(self._now())
        blocked_until = run.blocked_until_monotonic
        # The run's own status only advances when a call arrives, so an IDLE
        # process whose recovery window has already elapsed would report itself
        # dead while it is in fact ready to admit the next request as a probe. An
        # inspection surface must not report a state the engine would not act
        # on, so the snapshot reports what the next decision will see.
        #
        # "recovering" rather than "active", because the two are not the same
        # offer: an active run serves every caller, a recovering one serves
        # exactly one and refuses the rest until that one comes back. Reporting
        # it as active during an incident would hide an outage that is still
        # ongoing, which is the opposite of what this surface is read for.
        probe_due = self._probe_is_due(run)
        # A probe only counts as outstanding while the window armed for it is
        # still running. Past that the run is probe-due again whatever became of
        # it, so a ticket that was issued and never recorded cannot leave this
        # surface claiming a call is in flight for the rest of the process.
        probe_in_flight = run.probe_in_flight and not probe_due
        return {
            "run_id": run.run_id,
            "kind": run.kind,
            "status": (
                "recovering" if probe_due or probe_in_flight else run.status
            ),
            "mode": config.normalized_mode(),
            "enabled": config.enabled,
            "elapsed_seconds": run.elapsed_seconds(now),
            "total_calls": run.total_calls,
            "failed_calls": run.failed_calls,
            # Spend nobody consumed. Visible on its own line because otherwise a
            # disconnect storm reads as healthy traffic: calls climb, failures do
            # not, and nothing says the answers were thrown away.
            "cancelled_calls": run.cancelled_calls,
            # Cancellations are excluded from the denominator: they carry no
            # evidence about the provider, so counting them here would let a
            # disconnect storm dilute the displayed ratio and make an outage read
            # milder than it is. Enforcement never used this number (the windowed
            # ring already excludes cancellations), but an operator reads it.
            "failure_ratio": _ratio(
                run.failed_calls,
                run.total_calls - run.cancelled_calls,
            ),
            # Lifetime ratio above is ACCOUNTING; the windowed one below is what
            # the guard enforces on. They diverge on purpose once the run is
            # longer than the window, and the divergence is the point.
            "recent_calls": run.health.calls,
            "recent_failed_calls": run.health.failed,
            "recent_failure_ratio": run.health.failure_ratio(),
            "verdict_is_final": run.verdict_is_final,
            "blocked_seconds_remaining": (
                blocked_until - now if blocked_until is not None and not probe_due
                else None
            ),
            # How many health blocks this run has served without a probe clearing
            # them: 1 is a fresh trip, 5 is a provider that has been down through
            # five probes and is now being retried at the backoff ceiling.
            "health_block_streak": run.health_block_streak,
            "probe_in_flight": probe_in_flight,
            "recovery_count": run.recovery_count,
            "total_tokens": int(run.total_tokens),
            "reported_cost_usd": run.reported_cost_usd,
            # A token or cost budget measured in a quantity the provider never
            # reports cannot fire, and without these two lines that is
            # indistinguishable from a healthy run far below its ceiling.
            "successful_calls_without_token_usage": (
                run.successful_calls_without_token_usage
            ),
            "successful_calls_without_reported_cost": (
                run.successful_calls_without_reported_cost
            ),
            "total_latency_ms": run.total_latency_ms,
            "model_call_counts": dict(sorted(run.model_counts.items())),
            "error_class_counts": dict(sorted(run.error_counts.items())),
            # Kept while recovering: the evidence that blocked the run still
            # stands until a probe contradicts it, and an operator looking at a
            # recovering run needs to know what it is recovering FROM.
            "violations": list(run.violations),
            "thresholds": {
                "max_total_calls": config.max_total_calls,
                "max_total_failed_calls": config.max_total_failed_calls,
                "max_failed_call_ratio": config.max_failed_call_ratio,
                "min_calls_for_failed_ratio": config.min_calls_for_failed_ratio,
                "max_failed_calls_per_purpose": config.max_failed_calls_per_purpose,
                "max_failed_ratio_per_purpose": config.max_failed_ratio_per_purpose,
                "min_calls_per_purpose_for_failed_ratio": (
                    config.min_calls_per_purpose_for_failed_ratio
                ),
                "max_consecutive_failures_per_purpose": (
                    config.max_consecutive_failures_per_purpose
                ),
                "max_total_tokens": config.max_total_tokens,
                "max_reported_cost_usd": config.max_reported_cost_usd,
                "max_wall_time_seconds": config.max_wall_time_seconds,
                "health_window_calls": config.health_window_calls,
                "recovery_seconds": config.recovery_seconds,
            },
            "by_purpose": {
                purpose: {
                    "calls": counters.calls,
                    "failed_calls": counters.failed_calls,
                    "cancelled_calls": counters.cancelled_calls,
                    "latency_ms": counters.latency_ms,
                    # Same denominator rule as the run-level ratio above:
                    # evidence-bearing calls only.
                    "failure_ratio": _ratio(
                        counters.failed_calls,
                        counters.calls - counters.cancelled_calls,
                    ),
                    "recent_calls": counters.health.calls,
                    "recent_failed_calls": counters.health.failed,
                    "recent_failure_ratio": counters.health.failure_ratio(),
                    "consecutive_failures": counters.consecutive_failures,
                    "max_consecutive_failures": counters.max_consecutive_failures,
                }
                for purpose, counters in sorted(run.purposes.items())
            },
        }


# ---------------------------------------------------------------------------
# Per-turn LLM call meter (CS-1.4)
#
# The run guard above counts LLM calls process-wide (or per scoped bulk run) for
# budget/health enforcement, and only ONE run is active at a time (a scoped run
# replaces the runtime run). A single chat turn needs its OWN call count as a
# first-class trace metric that (a) does not disturb the process-wide guard
# accounting, (b) works whether or not a guard is configured, and (c) never
# influences an enforcement decision. The meter below is that separate,
# enforcement-free accumulator, threaded through a dedicated ContextVar.
#
# LLMClient records into the active meter at the same single provider-call
# choke point it already uses for the guard, so the meter cannot miss a call
# site and records exactly once per provider round-trip. Because it is a
# distinct accumulator, it never double-counts against the guard's run counters
# (read by admin/benchmark surfaces) nor against the benchmark's own outer-level
# LLMCallRecorder.
#
# ROUND-TRIP, NOT LOGICAL CALL. One logical client call can produce SEVERAL
# records: transient-error retries, the intimacy fallback, output-limit
# recovery, and the structured schema-drop fallback each re-enter the provider.
# That is the intended semantics for a cost/latency ratchet (every attempt
# spends money and wall time), and it is why a production trace reads higher
# than a stage count, and higher than the benchmark's LLMCallRecorder, which
# wraps the PUBLIC client methods and counts one per logical call.
#
# ATTEMPTED, NOT DELIVERED. A round-trip counts once it reaches the provider,
# whatever happens to its result: failures count, and so do calls abandoned by
# their caller (a proxy client disconnecting mid-stream). A counter that only
# saw delivered answers would report a disconnect storm as free.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LLMPurposeUsage:
    """Calls and provider latency accumulated for one purpose label."""

    calls: int = 0
    latency_ms: float = 0.0


@dataclass(slots=True)
class LLMCallMeter:
    """Mutable per-turn accumulator of provider LLM calls.

    ``by_purpose`` carries calls AND latency in one map on purpose (CS-1.5).
    Two parallel maps keyed by the same dimension are the shape that drifts:
    one gets a new label the other never sees. A single entry per purpose makes
    "which stage spent the wall time" answerable from the same record that
    answers "which stage spent the calls".
    """

    total_calls: int = 0
    failed_calls: int = 0
    total_latency_ms: float = 0.0
    by_purpose: dict[str, LLMPurposeUsage] = field(default_factory=dict)

    def record(
        self,
        *,
        purpose: str | None,
        latency_ms: float,
        outcome: LLMCallOutcome,
    ) -> None:
        """Record one provider round-trip that reached the provider.

        The meter measures SPEND, so every outcome counts as a call and adds its
        wall time -- a failed attempt and an abandoned stream both cost money and
        latency. ``failed_calls`` counts errors only: a cancelled round-trip is
        not an error, it is an answer nobody collected.
        """
        label = (purpose or "unknown").strip() or "unknown"
        normalized_latency_ms = max(0.0, float(latency_ms))
        usage = self.by_purpose.setdefault(label, LLMPurposeUsage())
        usage.calls += 1
        usage.latency_ms += normalized_latency_ms
        self.total_calls += 1
        self.total_latency_ms += normalized_latency_ms
        if outcome is LLMCallOutcome.FAILURE:
            self.failed_calls += 1


# Meters NEST. A proxy turn binds one meter for the whole turn and the sidecar
# retrieval it calls binds another for the retrieval scope alone; both must see
# the retrieval round-trips. A single-slot ContextVar would let the inner meter
# shadow the outer one and silently zero the turn count, so the bound value is
# the full stack and every recorded call fans out to all of it.
#
# UNBINDING IS BY IDENTITY, NOT BY TOKEN, and it is per-context and best-effort.
# ``ContextVar.set`` is context-local, so a scope that binds in one context and
# unbinds in another leaves the meter bound in the first one; ``Token.reset``
# would not fix that either (it is valid only in the Context that created the
# token, and it restores the whole tuple as of bind time, which can resurrect a
# meter another scope already removed). Mixing the two mechanisms is what made
# that possible, so there is exactly one: push a meter, remove that same object
# again. Correctness does not depend on the removal happening: a meter is only
# ever read by the code that created it, and a request's context dies with its
# task, so a meter left bound in a dying context is collected with it.
_CURRENT_CALL_METERS: ContextVar[tuple[LLMCallMeter, ...]] = ContextVar(
    "atagia_llm_call_meters",
    default=(),
)


def begin_llm_call_meter() -> LLMCallMeter:
    """Bind a fresh per-turn LLM call meter to the current async context."""
    meter = LLMCallMeter()
    bind_llm_call_meter(meter)
    return meter


def begin_isolated_llm_call_meter() -> LLMCallMeter:
    """Bind a fresh meter as the ONLY meter in the current context.

    For background work spawned from a turn. ``asyncio.create_task`` copies the
    spawning context, meters included, so a background task that merely PUSHES
    its own meter still charges every round-trip to the turn it came from --
    whose telemetry row is already written and will never be updated again.
    Background work belongs to no turn, so it replaces the inherited stack
    instead of extending it.

    Safe because the task runs in its own context copy: the spawning context's
    stack is untouched, and this meter dies with the task.
    """
    meter = LLMCallMeter()
    _CURRENT_CALL_METERS.set((meter,))
    return meter


def bind_llm_call_meter(meter: LLMCallMeter) -> None:
    """Push an existing meter onto the current context's meter stack.

    Used when one logical turn spans two async scopes (the proxy's streaming
    path: setup coroutine, then the response generator) and both scopes must
    accumulate into the same meter.

    Binding the SAME meter twice in one context would make every subsequent
    round-trip land on it twice, since ``record_call_on_active_meter`` fans out
    over the whole stack. The streaming proxy is safe only because the
    generator's context is created after the setup scope unbinds, which is a
    property of how the scopes nest rather than anything the API enforces, so
    enforce it here: a double bind is a caller bug, not a tolerable state.
    """
    bound = _CURRENT_CALL_METERS.get()
    if any(existing is meter for existing in bound):
        raise RuntimeError(
            "LLM call meter is already bound in this context; binding it again "
            "would double-count every subsequent provider round-trip"
        )
    _CURRENT_CALL_METERS.set((*bound, meter))


def end_llm_call_meter(meter: LLMCallMeter) -> None:
    """Remove ``meter`` from the current context's stack, by object identity.

    Order-independent: ending an outer meter while an inner one is still bound
    leaves the inner one bound, which is what a nested proxy/sidecar turn needs.
    """
    _CURRENT_CALL_METERS.set(
        tuple(bound for bound in _CURRENT_CALL_METERS.get() if bound is not meter)
    )


def record_call_on_active_meter(
    *,
    purpose: str | None,
    latency_ms: float,
    outcome: LLMCallOutcome,
) -> None:
    """Record one provider round-trip into every bound meter."""
    for meter in _CURRENT_CALL_METERS.get():
        meter.record(purpose=purpose, latency_ms=latency_ms, outcome=outcome)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _total_tokens(usage: dict[str, Any]) -> float:
    total = _number_at(usage, ("total_tokens",)) or _number_at(usage, ("totalTokenCount",))
    if total is not None:
        return total
    input_tokens = (
        _number_at(usage, ("input_tokens",))
        or _number_at(usage, ("prompt_tokens",))
        or _number_at(usage, ("promptTokenCount",))
        or 0.0
    )
    output_tokens = (
        _number_at(usage, ("output_tokens",))
        or _number_at(usage, ("completion_tokens",))
        or _number_at(usage, ("candidatesTokenCount",))
        or 0.0
    )
    return input_tokens + output_tokens


def _reported_cost(usage: dict[str, Any]) -> float:
    return (
        _number_at(usage, ("cost",))
        or _number_at(usage, ("cost_details", "upstream_inference_cost"))
        or 0.0
    )


def _number_at(source: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = source
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None
