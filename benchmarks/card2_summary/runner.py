"""Run the card 2 summary harness against a Compactor, with instrumentation.

The harness drives the real engine method ``_summarize_message_ranges_card``.
To count calls, time them, observe concurrency, and capture errors WITHOUT
touching the engine prompt, every provider used here is wrapped in
``_CountingProvider``: it delegates ``complete()`` unchanged but, for the
per-range summary purpose, increments/decrements an in-flight counter (recording
the max), counts calls, and records exceptions. The engine never sees a modified
request or response.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.db_sqlite import initialize_database
from atagia.core.repositories import (
    ConversationRepository,
    UserRepository,
    WorkspaceRepository,
)
from atagia.memory.compactor import Compactor
from atagia.memory.policy_manifest import ManifestLoader, sync_assistant_modes
from atagia.services.llm_client import (
    LLMClient,
    LLMCompletionRequest,
    LLMCompletionResponse,
    LLMEmbeddingRequest,
    LLMEmbeddingResponse,
    LLMProvider,
    LLMRequestError,
    TransientLLMError,
)

from benchmarks.card2_summary.cases import (
    FROZEN_RANGE_FIXTURE_ID,
    Card2Case,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS_DIR = _REPO_ROOT / "migrations"
MANIFESTS_DIR = _REPO_ROOT / "manifests"

_SUMMARY_PURPOSE = "summary_chunk_segmentation_summaries_card"


class _CountingProvider(LLMProvider):
    """Wrap a provider to instrument the per-range summary card only.

    Counts summary ``complete()`` calls, tracks max concurrency via an in-flight
    counter, times each call, and records exceptions by type/message. The wrapped
    provider's ``complete()`` runs unchanged, so the engine prompt and behavior
    are untouched.
    """

    def __init__(self, inner: LLMProvider) -> None:
        self._inner = inner
        self.name = inner.name
        self.supports_embeddings = inner.supports_embeddings
        self.supports_embedding_dimensions = inner.supports_embedding_dimensions
        self.supports_native_structured_output = (
            inner.supports_native_structured_output
        )
        self.summary_call_count = 0
        self.max_concurrent_summaries = 0
        self._active_summaries = 0
        self._lock = asyncio.Lock()
        # (error_type, message, retryable) tuples captured from summary calls.
        self.errors: list[tuple[str, str, bool]] = []

    async def complete(self, request: LLMCompletionRequest) -> LLMCompletionResponse:
        is_summary = str(request.metadata.get("purpose")) == _SUMMARY_PURPOSE
        if not is_summary:
            return await self._inner.complete(request)
        async with self._lock:
            self.summary_call_count += 1
            self._active_summaries += 1
            if self._active_summaries > self.max_concurrent_summaries:
                self.max_concurrent_summaries = self._active_summaries
        try:
            return await self._inner.complete(request)
        except Exception as exc:  # noqa: BLE001 -- record every failure type
            retryable = isinstance(exc, TransientLLMError)
            self.errors.append((type(exc).__name__, str(exc), retryable))
            raise
        finally:
            async with self._lock:
                self._active_summaries -= 1

    async def embed(self, request: LLMEmbeddingRequest) -> LLMEmbeddingResponse:
        return await self._inner.embed(request)

    def supports_native_structured_output_for(
        self, request: LLMCompletionRequest
    ) -> bool:
        return self._inner.supports_native_structured_output_for(request)


@dataclass(slots=True)
class CaseMetrics:
    case_id: str
    family: str
    range_count: int
    summaries_returned: int
    range_coverage: float
    retry_count: int
    hard_failure: bool
    hard_failure_error: str | None
    wall_clock_ms: float
    llm_call_count: int
    observed_max_concurrency: int
    rate_limit_error_count: int
    transient_error_count: int
    request_error_count: int
    errors: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class RunReport:
    model: str
    concurrency: int
    runs: int
    case_set: str
    card_examples_enabled: bool
    frozen_range_fixture_id: str
    selftest: bool
    generated_at_utc: str
    per_case: list[CaseMetrics] = field(default_factory=list)
    aggregate: dict[str, Any] = field(default_factory=dict)


def build_settings(
    *,
    model: str,
    concurrency: int,
    card_examples_enabled: bool,
    base: Settings | None = None,
) -> Settings:
    """Build Settings that route the compactor to ``model`` at ``concurrency``.

    Concurrency must be >= 1 (Settings rejects <= 0); the engine runs sequentially
    when the cap is 1.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    template = base or _default_settings()
    overrides = asdict(template)
    overrides["compactor_summary_card_concurrency"] = concurrency
    overrides["card_examples_enabled"] = card_examples_enabled
    overrides["llm_component_models"] = {
        **dict(template.llm_component_models),
        "compactor": model,
    }
    return Settings(**overrides)


def _default_settings() -> Settings:
    """A minimal, env-independent Settings template for the harness."""
    return Settings(
        sqlite_path=":memory:",
        migrations_path=str(MIGRATIONS_DIR),
        manifests_path=str(MANIFESTS_DIR),
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        openai_api_key="harness-openai-placeholder",
        openrouter_api_key=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
        llm_chat_model="harness-chat-model",
        service_mode=False,
        service_api_key=None,
        admin_api_key=None,
        workers_enabled=False,
        debug=False,
        allow_insecure_http=True,
    )


async def _build_compactor(
    *,
    settings: Settings,
    provider: _CountingProvider,
    provider_name: str | None,
) -> tuple[Any, Compactor]:
    """Build an in-memory Compactor wired to ``provider`` (mirrors test_compactor)."""
    connection = await initialize_database(":memory:", MIGRATIONS_DIR)
    clock = FrozenClock(datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc))
    await sync_assistant_modes(
        connection, ManifestLoader(MANIFESTS_DIR).load_all(), clock
    )
    users = UserRepository(connection, clock)
    workspaces = WorkspaceRepository(connection, clock)
    conversations = ConversationRepository(connection, clock)
    await users.create_user("usr_card2")
    await workspaces.create_workspace("wrk_card2", "usr_card2", "Card2 Harness")
    await conversations.create_conversation(
        "cnv_card2", "usr_card2", "wrk_card2", "coding_debug", "Card2"
    )
    client: LLMClient[Any] = LLMClient(
        provider_name=provider_name,
        providers=[provider],
    )
    compactor = Compactor(
        connection=connection,
        llm_client=client,
        clock=clock,
        settings=settings,
    )
    return connection, compactor


async def _run_case(
    *,
    compactor: Compactor,
    provider: _CountingProvider,
    case: Card2Case,
) -> CaseMetrics:
    provider.summary_call_count = 0
    provider.max_concurrent_summaries = 0
    provider.errors.clear()

    range_count = len(case.ranges)
    started = perf_counter()
    hard_failure = False
    hard_failure_error: str | None = None
    summaries_by_range: dict[tuple[int, int], str] = {}
    try:
        summaries_by_range = await compactor._summarize_message_ranges_card(
            user_id="usr_card2",
            conversation_id="cnv_card2",
            messages=case.messages,
            ranges=case.ranges,
        )
    except Exception as exc:  # noqa: BLE001 -- a raised range after retries = hard failure
        hard_failure = True
        hard_failure_error = f"{type(exc).__name__}: {exc}"
    wall_clock_ms = (perf_counter() - started) * 1000.0

    summaries_returned = sum(
        1 for value in summaries_by_range.values() if value and value.strip()
    )
    range_coverage = (summaries_returned / range_count) if range_count else 0.0
    # One call per range is the floor; everything above it is a retry.
    retry_count = max(0, provider.summary_call_count - range_count)

    rate_limit_error_count = 0
    transient_error_count = 0
    request_error_count = 0
    errors: list[dict[str, Any]] = []
    for error_type, message, retryable in provider.errors:
        errors.append(
            {"type": error_type, "message": message[:300], "retryable": retryable}
        )
        if retryable:
            transient_error_count += 1
        if "429" in message or "rate limit" in message.lower():
            rate_limit_error_count += 1
        if error_type == LLMRequestError.__name__:
            request_error_count += 1

    return CaseMetrics(
        case_id=case.case_id,
        family=case.family,
        range_count=range_count,
        summaries_returned=summaries_returned,
        range_coverage=range_coverage,
        retry_count=retry_count,
        hard_failure=hard_failure,
        hard_failure_error=hard_failure_error,
        wall_clock_ms=wall_clock_ms,
        llm_call_count=provider.summary_call_count,
        observed_max_concurrency=provider.max_concurrent_summaries,
        rate_limit_error_count=rate_limit_error_count,
        transient_error_count=transient_error_count,
        request_error_count=request_error_count,
        errors=errors,
    )


async def run_harness(
    *,
    cases: list[Card2Case],
    model: str,
    concurrency: int,
    runs: int,
    case_set: str,
    inner_provider: LLMProvider,
    provider_name: str | None,
    card_examples_enabled: bool,
    selftest: bool,
    base_settings: Settings | None = None,
) -> RunReport:
    """Run every case ``runs`` times and return aggregated metrics.

    ``inner_provider`` is the real (or fake) provider; it is wrapped here so the
    same Compactor instance is reused across runs and cases, accumulating no
    cross-case state beyond what each case resets.
    """
    settings = build_settings(
        model=model,
        concurrency=concurrency,
        card_examples_enabled=card_examples_enabled,
        base=base_settings,
    )
    provider = _CountingProvider(inner_provider)
    connection, compactor = await _build_compactor(
        settings=settings,
        provider=provider,
        provider_name=provider_name,
    )
    per_case: list[CaseMetrics] = []
    try:
        for run_index in range(runs):
            for case in cases:
                metrics = await _run_case(
                    compactor=compactor,
                    provider=provider,
                    case=case,
                )
                metrics.case_id = (
                    metrics.case_id if runs == 1 else f"{case.case_id}#run{run_index + 1}"
                )
                per_case.append(metrics)
    finally:
        await connection.close()

    report = RunReport(
        model=model,
        concurrency=concurrency,
        runs=runs,
        case_set=case_set,
        card_examples_enabled=card_examples_enabled,
        frozen_range_fixture_id=FROZEN_RANGE_FIXTURE_ID,
        selftest=selftest,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        per_case=per_case,
    )
    report.aggregate = _aggregate(per_case)
    return report


def _aggregate(per_case: list[CaseMetrics]) -> dict[str, Any]:
    if not per_case:
        return {}
    total_ranges = sum(metrics.range_count for metrics in per_case)
    total_summaries = sum(metrics.summaries_returned for metrics in per_case)
    return {
        "case_count": len(per_case),
        "total_ranges": total_ranges,
        "total_summaries_returned": total_summaries,
        "overall_range_coverage": (
            total_summaries / total_ranges if total_ranges else 0.0
        ),
        "total_retry_count": sum(metrics.retry_count for metrics in per_case),
        "hard_failure_case_count": sum(
            1 for metrics in per_case if metrics.hard_failure
        ),
        "total_llm_call_count": sum(metrics.llm_call_count for metrics in per_case),
        "observed_max_concurrency": max(
            metrics.observed_max_concurrency for metrics in per_case
        ),
        "total_wall_clock_ms": sum(metrics.wall_clock_ms for metrics in per_case),
        "rate_limit_error_count": sum(
            metrics.rate_limit_error_count for metrics in per_case
        ),
        "transient_error_count": sum(
            metrics.transient_error_count for metrics in per_case
        ),
        "request_error_count": sum(
            metrics.request_error_count for metrics in per_case
        ),
    }
