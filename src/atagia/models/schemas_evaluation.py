"""Schemas for evaluation metrics and dashboard responses."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from atagia.models.schemas_memory import TurnSurface


class MetricName(str, Enum):
    """Supported dashboard metric identifiers."""

    MUR = "mur"
    IPR = "ipr"
    SLR = "slr"
    BDER = "bder"
    CCR = "ccr"
    SYSTEM = "system"


class MetricResult(BaseModel):
    """Aggregate metric value plus the contributing sample count."""

    model_config = ConfigDict(extra="forbid")

    value: float
    sample_count: int = Field(ge=0)


class EvaluationMetricRecord(BaseModel):
    """Persisted aggregate metric row."""

    model_config = ConfigDict(extra="forbid")

    id: str
    metric_name: str
    metric_value: float
    sample_count: int = Field(ge=0)
    user_id: str | None = None
    assistant_mode_id: str | None = None
    workspace_id: str | None = None
    time_bucket: str
    computed_at: datetime


class ContractComplianceEvaluation(BaseModel):
    """Structured LLM judgment for contract compliance."""

    model_config = ConfigDict(extra="ignore")

    compliance_score: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1)


class RetrievalEventMeasurements(BaseModel):
    """Aggregate measurements over one slice of raw retrieval events.

    Latency is reported as THREE series that are never averaged together,
    because they measure different physical quantities and a mean over them
    would be none of the three:

    * ``avg_retrieval_stage_latency_ms`` -- the retrieval stage timing itself,
      over the events that persisted a ``retrieval_duration_ms`` for a
      retrieval that produced memory context.
    * ``avg_failed_retrieval_stage_latency_ms`` -- the same measured stage
      timing, over the events whose retrieval failed open (the proxy's fallback
      row, which records ``memory_context_available=false``). The spend was
      real and stays visible here, but a failed attempt does not describe what a
      working retrieval costs, so it never enters the series above.
    * ``avg_request_to_event_wall_ms`` -- the wall gap between the request
      message landing and the event row being written, over the events written
      before the turn-telemetry migration, which are the only ones without a
      measured stage duration. That gap also contains the reply generation, so
      it bounds a turn from above and is not a retrieval-stage figure.

    Each series carries its own sample count, which is exactly the population
    behind its figure. No event feeds both: an event that persisted a stage
    duration contributes to the measured series only. An average whose count is
    0 is reported as 0.0 and means "nothing in this slice contributed", never
    "the measurement was zero" -- check the count before displaying.
    """

    model_config = ConfigDict(extra="forbid")

    total_events: int = Field(ge=0)
    cold_start_count: int = Field(ge=0)
    zero_candidate_count: int = Field(ge=0)
    avg_items_included: float
    avg_items_dropped: float
    avg_token_estimate: float
    avg_retrieval_stage_latency_ms: float
    retrieval_stage_latency_sample_count: int = Field(ge=0)
    avg_failed_retrieval_stage_latency_ms: float
    failed_retrieval_stage_latency_sample_count: int = Field(ge=0)
    avg_request_to_event_wall_ms: float
    request_to_event_wall_sample_count: int = Field(ge=0)


class RetrievalSummaryStats(RetrievalEventMeasurements):
    """Direct summary over raw retrieval events, split by turn surface.

    The inherited top-level fields cover every event the filters matched, which
    mixes surfaces that are not comparable: a ``chat`` or proxy event covers a
    whole turn including the reply call, while a ``context`` event covers
    retrieval only because the host runs its own model. ``by_surface`` carries
    the same measurements per surface so a blended number cannot be read as a
    chat number, and ``surface_filter`` echoes the requested restriction
    (``None`` means every surface). ``by_surface`` only holds surfaces actually
    present in the window; absent surfaces are omitted rather than reported as
    zero, which would fabricate measurements nobody took.
    """

    surface_filter: TurnSurface | None
    by_surface: dict[TurnSurface, RetrievalEventMeasurements]
