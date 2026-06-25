"""Dataset models and loader for the adaptive retrieval gate suite.

The suite is built from synthetic, fictional content only. Each persona owns a
short setup conversation (the memory the engine must store) plus a set of probe
questions. Probe questions come in two flavors:

* ``paired`` probes share a ``pair_id`` and cover the same topic in two
  variants: a ``world`` variant answerable from parametric knowledge alone, and
  a ``personal`` variant that depends on the user's stored memory. The pairing
  is what makes a false-skip (skipping the personal variant) directly
  comparable to a correct-skip (skipping the world variant) on the same topic.
* ``conversation`` probes ask about something the engine can answer from the
  visible recent window alone.

Every question is labeled with the expected :class:`MemoryDependence` and the
expected gate action, so classification accuracy and end-to-end behavior can be
scored without any benchmark-aware engine logic.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from atagia.models.schemas_memory import MemoryDependence


_DATA_DIR = Path(__file__).resolve().parent / "data"
_DEFAULT_DATASET_FILE = _DATA_DIR / "gate_suite_v0.json"


class GateExpectedAction(BaseModel):
    """Expected gate behavior for one probe question.

    ``skip`` means the gate should short-circuit retrieval (the
    contract-only path); ``retrieve`` means the gate must keep full retrieval.
    The action is DERIVED from the expected :class:`MemoryDependence` by
    :attr:`GateProbeQuestion.expected_action` (``world``/``conversation`` skip;
    ``personal``/``mixed`` retrieve). The dataset declares only
    ``expected_memory_dependence`` -- the single source of truth -- so there is
    no separately declared action to validate against.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    skip_retrieval: bool


class GateProbeTurn(BaseModel):
    """One turn in a gate-suite setup conversation."""

    model_config = ConfigDict(extra="forbid")

    turn_id: str
    role: str
    text: str
    timestamp: str


class GateProbeConversation(BaseModel):
    """A short setup conversation that seeds the engine's memory."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    assistant_mode_id: str
    timestamp_base: str
    turns: list[GateProbeTurn] = Field(default_factory=list)


class GateProbeQuestion(BaseModel):
    """One gate probe question with its expected classification and action.

    ``language`` is an informational tag (BCP-47-ish) describing the language
    the question is written in; the engine never reads it. It exists so the
    suite can assert multilingual coverage and so reports can slice accuracy by
    language.
    """

    model_config = ConfigDict(extra="forbid")

    question_id: str
    question_text: str
    language: str = "en"
    probe_kind: str
    pair_id: str | None = None
    target_conversation_id: str
    assistant_mode_id: str
    expected_memory_dependence: MemoryDependence
    notes: str = ""

    @property
    def expected_action(self) -> GateExpectedAction:
        """Return the gate action implied by the expected dependence.

        ``world`` and ``conversation`` turns should skip retrieval;
        ``personal`` and ``mixed`` turns must retrieve. This mirrors the gate's
        conservative semantics (uncertainty -> retrieve) without encoding any
        benchmark-specific branch.
        """
        skip = self.expected_memory_dependence in (
            MemoryDependence.WORLD,
            MemoryDependence.CONVERSATION,
        )
        return GateExpectedAction(skip_retrieval=skip)


class GateProbePersona(BaseModel):
    """All data for one gate-suite persona."""

    model_config = ConfigDict(extra="forbid")

    persona_id: str
    display_name: str
    profile: str
    conversations: list[GateProbeConversation] = Field(default_factory=list)
    questions: list[GateProbeQuestion] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_question_targets(self) -> "GateProbePersona":
        """Fail fast when a question targets an unknown setup conversation."""
        known = {conversation.conversation_id for conversation in self.conversations}
        for question in self.questions:
            if question.target_conversation_id not in known:
                raise ValueError(
                    f"Question {question.question_id} targets unknown "
                    f"conversation {question.target_conversation_id!r}"
                )
        return self


class GateSuiteDataset(BaseModel):
    """Complete adaptive retrieval gate suite."""

    model_config = ConfigDict(extra="forbid")

    name: str = "atagia-bench-gate-v0"
    personas: list[GateProbePersona] = Field(default_factory=list)

    @property
    def total_questions(self) -> int:
        """Return the number of probe questions across all personas."""
        return sum(len(persona.questions) for persona in self.personas)

    @property
    def total_conversations(self) -> int:
        """Return the number of setup conversations across all personas."""
        return sum(len(persona.conversations) for persona in self.personas)

    @property
    def questions(self) -> list[GateProbeQuestion]:
        """Return every probe question across all personas in order."""
        return [
            question
            for persona in self.personas
            for question in persona.questions
        ]

    @property
    def languages(self) -> set[str]:
        """Return the set of languages present across all probe questions."""
        return {question.language for question in self.questions}

    @model_validator(mode="after")
    def _validate_pairs(self) -> "GateSuiteDataset":
        """Fail fast on malformed pairing.

        A ``pair_id`` must group exactly one ``world`` variant and one
        ``personal`` variant, so false-skip and correct-skip are comparable on
        the same topic. Question ids must be globally unique.
        """
        seen_ids: set[str] = set()
        pairs: dict[str, list[GateProbeQuestion]] = {}
        for question in self.questions:
            if question.question_id in seen_ids:
                raise ValueError(
                    f"Duplicate question id: {question.question_id}"
                )
            seen_ids.add(question.question_id)
            if question.pair_id is not None:
                pairs.setdefault(question.pair_id, []).append(question)

        for pair_id, members in pairs.items():
            dependences = sorted(
                member.expected_memory_dependence.value for member in members
            )
            if dependences != ["personal", "world"]:
                raise ValueError(
                    f"Pair {pair_id!r} must contain exactly one 'world' and one "
                    f"'personal' variant, found: {dependences}"
                )
        return self


def load_gate_suite(dataset_file: str | Path | None = None) -> GateSuiteDataset:
    """Load and validate the gate suite from its JSON dataset file."""
    path = Path(dataset_file) if dataset_file else _DEFAULT_DATASET_FILE
    raw = json.loads(path.read_text(encoding="utf-8"))
    return GateSuiteDataset.model_validate(raw)
