"""Shared benchmark schemas and extension points."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


UNSCORED_CATEGORY = 5
DEFAULT_SCORED_CATEGORIES = [1, 2, 3, 4]


class BenchmarkTurn(BaseModel):
    """One normalized conversation turn."""

    model_config = ConfigDict(extra="forbid")

    role: str
    text: str
    speaker: str
    session_id: str
    timestamp: str
    turn_id: str | None = None


class BenchmarkQuestion(BaseModel):
    """One benchmark question associated with a conversation."""

    model_config = ConfigDict(extra="forbid")

    question_text: str
    ground_truth: str
    category: int = Field(ge=1)
    evidence_turn_ids: list[str] = Field(default_factory=list)
    question_id: str

    @property
    def is_scored(self) -> bool:
        """Return whether this question should contribute to benchmark accuracy."""
        return self.category != UNSCORED_CATEGORY


class BenchmarkConversation(BaseModel):
    """One benchmark conversation with normalized turns and questions."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    turns: list[BenchmarkTurn] = Field(default_factory=list)
    questions: list[BenchmarkQuestion] = Field(default_factory=list)

    @property
    def scored_questions(self) -> list[BenchmarkQuestion]:
        """Return all scored questions for the conversation."""
        return [question for question in self.questions if question.is_scored]

    def filtered_questions(self, categories: list[int] | None = None) -> list[BenchmarkQuestion]:
        """Return scored questions constrained to the requested categories."""
        if categories is None:
            return self.scored_questions
        allowed_categories = set(categories)
        return [
            question
            for question in self.scored_questions
            if question.category in allowed_categories
        ]


class BenchmarkDataset(BaseModel):
    """A benchmark dataset containing multiple conversations."""

    model_config = ConfigDict(extra="forbid")

    name: str
    conversations: list[BenchmarkConversation] = Field(default_factory=list)


class ScoreResult(BaseModel):
    """Judge output for one predicted answer."""

    model_config = ConfigDict(extra="forbid")

    score: int = Field(ge=0, le=1)
    reasoning: str
    judge_model: str


class QuestionResult(BaseModel):
    """Full output for one benchmarked question."""

    model_config = ConfigDict(extra="forbid")

    question: BenchmarkQuestion
    prediction: str
    score_result: ScoreResult
    memories_used: int = Field(ge=0)
    retrieval_time_ms: float = Field(ge=0.0)


class ConversationReport(BaseModel):
    """Aggregated results for one benchmark conversation."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    results: list[QuestionResult] = Field(default_factory=list)
    accuracy: float = Field(ge=0.0, le=1.0)
    category_breakdown: dict[int, float] = Field(default_factory=dict)


class BenchmarkReport(BaseModel):
    """Aggregated report for a full benchmark run."""

    model_config = ConfigDict(extra="forbid")

    benchmark_name: str
    overall_accuracy: float = Field(ge=0.0, le=1.0)
    category_breakdown: dict[int, float] = Field(default_factory=dict)
    conversations: list[ConversationReport] = Field(default_factory=list)
    total_questions: int = Field(ge=0)
    total_correct: int = Field(ge=0)
    ablation_config: dict[str, Any] | None = None
    timestamp: str
    model_info: dict[str, Any] = Field(default_factory=dict)
    duration_seconds: float = Field(ge=0.0)


class BenchmarkAdapter(ABC):
    """Abstract loader for benchmark datasets."""

    @abstractmethod
    def load(self) -> BenchmarkDataset:
        """Load and normalize the benchmark dataset."""


class BenchmarkRunner(ABC):
    """Abstract benchmark runner."""

    @abstractmethod
    async def run(self, *args: Any, **kwargs: Any) -> BenchmarkReport:
        """Execute the benchmark and return a report."""
