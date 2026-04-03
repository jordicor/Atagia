"""Schemas for evaluation metrics and dashboard responses."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


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

    model_config = ConfigDict(extra="forbid")

    compliance_score: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1)


class RetrievalSummaryStats(BaseModel):
    """Direct summary over raw retrieval events."""

    model_config = ConfigDict(extra="forbid")

    total_events: int = Field(ge=0)
    cold_start_count: int = Field(ge=0)
    zero_candidate_count: int = Field(ge=0)
    avg_items_included: float
    avg_items_dropped: float
    avg_token_estimate: float
    avg_retrieval_latency_ms: float
