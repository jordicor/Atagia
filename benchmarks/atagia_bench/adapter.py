"""Adapter for loading the Atagia-bench dataset from authored JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


_DATA_DIR = Path(__file__).resolve().parent / "data"


# ---- Atagia-bench specific models ----


class AtagiaBenchPersona(BaseModel):
    """Persona definition loaded from personas.json."""

    model_config = ConfigDict(extra="forbid")

    persona_id: str
    display_name: str
    age: int
    occupation: str
    profile: str
    assistant_modes: list[str]
    conversation_count: int
    test_scenarios: list[str]


class AtagiaBenchTurn(BaseModel):
    """One conversation turn in an authored conversation."""

    model_config = ConfigDict(extra="forbid")

    turn_id: str
    role: str
    text: str
    timestamp: str


class AtagiaBenchConversation(BaseModel):
    """One authored conversation belonging to a persona."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    assistant_mode_id: str
    timestamp_base: str
    turns: list[AtagiaBenchTurn] = Field(default_factory=list)


class AtagiaBenchQuestion(BaseModel):
    """One benchmark question with ground truth and grader config."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    question_text: str
    ground_truth: str
    answer_type: str
    category_tags: list[str] = Field(default_factory=list)
    evidence_turn_ids: list[str] = Field(default_factory=list)
    grader: str
    grader_config: dict[str, Any] | None = None


class AtagiaBenchPersonaData(BaseModel):
    """All data for one persona: conversations and questions."""

    model_config = ConfigDict(extra="forbid")

    persona: AtagiaBenchPersona
    conversations: list[AtagiaBenchConversation] = Field(default_factory=list)
    questions: list[AtagiaBenchQuestion] = Field(default_factory=list)


class AtagiaBenchDataset(BaseModel):
    """Complete Atagia-bench dataset."""

    model_config = ConfigDict(extra="forbid")

    name: str = "atagia-bench-v0"
    personas: list[AtagiaBenchPersonaData] = Field(default_factory=list)

    @property
    def total_questions(self) -> int:
        return sum(len(p.questions) for p in self.personas)

    @property
    def total_conversations(self) -> int:
        return sum(len(p.conversations) for p in self.personas)


# ---- Loader ----


class AtagiaBenchAdapter:
    """Load the Atagia-bench dataset from the data/ directory."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self._data_dir = Path(data_dir) if data_dir else _DATA_DIR

    @property
    def data_dir(self) -> Path:
        """Return the dataset directory used by this adapter."""
        return self._data_dir

    def load(
        self,
        persona_ids: list[str] | None = None,
    ) -> AtagiaBenchDataset:
        """Load and return the full dataset, optionally filtered by persona."""
        personas_path = self._data_dir / "personas.json"
        raw_personas = json.loads(personas_path.read_text(encoding="utf-8"))
        all_personas = [
            AtagiaBenchPersona.model_validate(entry)
            for entry in raw_personas
        ]

        if persona_ids is not None:
            requested = set(persona_ids)
            all_personas = [p for p in all_personas if p.persona_id in requested]
            missing = requested - {p.persona_id for p in all_personas}
            if missing:
                raise ValueError(
                    f"Unknown persona ids: {', '.join(sorted(missing))}"
                )

        persona_data: list[AtagiaBenchPersonaData] = []
        for persona in all_personas:
            persona_dir = self._data_dir / persona.persona_id
            conversations = self._load_conversations(persona_dir)
            questions = self._load_questions(persona_dir)
            persona_data.append(
                AtagiaBenchPersonaData(
                    persona=persona,
                    conversations=conversations,
                    questions=questions,
                )
            )

        return AtagiaBenchDataset(personas=persona_data)

    def _load_conversations(
        self,
        persona_dir: Path,
    ) -> list[AtagiaBenchConversation]:
        """Load conversations for a single persona."""
        conversations_path = persona_dir / "conversations.json"
        if not conversations_path.exists():
            raise FileNotFoundError(
                f"Missing conversations file: {conversations_path}"
            )
        raw = json.loads(conversations_path.read_text(encoding="utf-8"))
        return [
            AtagiaBenchConversation.model_validate(entry)
            for entry in raw
        ]

    def _load_questions(
        self,
        persona_dir: Path,
    ) -> list[AtagiaBenchQuestion]:
        """Load questions for a single persona."""
        questions_path = persona_dir / "questions.json"
        if not questions_path.exists():
            raise FileNotFoundError(
                f"Missing questions file: {questions_path}"
            )
        raw = json.loads(questions_path.read_text(encoding="utf-8"))
        return [
            AtagiaBenchQuestion.model_validate(entry)
            for entry in raw
        ]
