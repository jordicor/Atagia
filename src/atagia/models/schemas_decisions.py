"""Typed, independent decisions consumed directly by application code."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, field_validator


class ChoiceQuestion(BaseModel):
    """One finite-choice question; never a free-text completion."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["choice"] = "choice"
    instructions: str = Field(min_length=1)
    criteria: dict[str, str | None] = Field(min_length=2, max_length=255)


class ChoiceAnswer(BaseModel):
    """A selected option and its provider-reported distribution."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["choice"]
    choice: str
    probabilities: dict[str, float]
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)


class ScoreQuestion(BaseModel):
    """Five ordered levels for the memory-source support rubric."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["score"] = "score"
    instructions: str = Field(min_length=1)
    criteria: tuple[str, str, str, str, str]

    @field_validator("criteria")
    @classmethod
    def require_described_levels(cls, levels: tuple[str, ...]) -> tuple[str, ...]:
        if any(not level.strip() for level in levels):
            raise ValueError("Every score level needs a description")
        return levels


class ScoreAnswer(BaseModel):
    """Raw weighted level, legend, distribution and provider certainty."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["score"]
    score: StrictFloat = Field(ge=0, le=4, allow_inf_nan=False)
    legend: dict[str, str]
    probabilities: dict[str, StrictFloat]
    confidence: StrictFloat = Field(ge=0, le=1, allow_inf_nan=False)

    @property
    def normalized_score(self) -> float:
        """Map the weighted five-level result to memory confidence in [0, 1]."""
        return self.score / 4


class ScoreDecision(BaseModel):
    """Common scalar result; typed details remain separate when available."""

    model_config = ConfigDict(extra="forbid")

    normalized_score: float = Field(ge=0, le=1, allow_inf_nan=False)
    typed_answer: ScoreAnswer | None = None
