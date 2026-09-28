"""Shared turn-telemetry payloads for tests that seed retrieval events.

Most tests that create a retrieval event care about identity, scoping, or
replay, not about latency numbers. They still have to supply the required
telemetry argument, so they use one fixed payload from here instead of
inventing their own.
"""

from __future__ import annotations

from atagia.core.retrieval_event_repository import TurnTelemetry
from atagia.models.schemas_memory import LLMPurposeMetricsTrace, TurnSurface
from atagia.services.llm_run_guard import (
    LLMCallMeter,
    begin_isolated_llm_call_meter,
    begin_llm_call_meter,
    bind_llm_call_meter,
    end_llm_call_meter,
)


class TurnCallMeterMixin:
    """Per-turn meter surface for stub LLM clients used in proxy tests.

    Stub clients replace ``runtime.llm_client`` wholesale, and a turn binds its
    meter through the client, so a stub without these would break the turn. The
    real client's methods are the same one-line delegations.
    """

    def begin_turn_call_meter(self) -> LLMCallMeter:
        return begin_llm_call_meter()

    def begin_isolated_call_meter(self) -> LLMCallMeter:
        return begin_isolated_llm_call_meter()

    def bind_turn_call_meter(self, meter: LLMCallMeter) -> None:
        bind_llm_call_meter(meter)

    def end_turn_call_meter(self, meter: LLMCallMeter) -> None:
        end_llm_call_meter(meter)


def sample_turn_telemetry(
    surface: TurnSurface = TurnSurface.CHAT,
) -> TurnTelemetry:
    """Return a fixed, valid telemetry payload for a seeded retrieval event."""
    return TurnTelemetry(
        surface=surface,
        turn_to_event_write_wall_ms=25.0,
        retrieval_duration_ms=10.0,
        llm_total_calls=3,
        llm_failed_calls=1,
        llm_total_latency_ms=18.5,
        llm_by_purpose={
            "need_detection": LLMPurposeMetricsTrace(calls=2, latency_ms=12.5),
            "chat_reply": LLMPurposeMetricsTrace(calls=1, latency_ms=6.0),
        },
        stage_timings_ms={"planning": 4.0, "candidate_search": 6.0},
    )
