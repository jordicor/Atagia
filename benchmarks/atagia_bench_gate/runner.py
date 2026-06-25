"""Runner for the adaptive retrieval gate diagnostic suite.

The runner ingests each persona's short setup conversation through a real
in-memory Atagia engine, then asks every probe question with the adaptive
retrieval gate enabled. For each question it captures what the gate actually
did (classification + skip/retrieve action) from the chat-result debug payload
and scores it against the expected labels.

Enabling the gate: the gate is a per-turn behavior controlled by the
``adaptive_retrieval`` setting / request override (see the feature config
lever). This runner enables it through the ``ATAGIA_ADAPTIVE_RETRIEVAL``
environment variable for the engine default and also passes the per-request
``adaptive_retrieval`` flag on ``Atagia.chat``. The runner itself contains no
gate logic and no benchmark-specific branches: it only reads the observable
trace the engine produces.

The suite is a diagnostic, not an accuracy gate: classification accuracy is
scorable from shadow-mode traces alone, while skip/retrieve action accuracy is
scorable only once the engine reports a concrete gate status. Whatever the
engine does not report stays unscored rather than guessed.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

from pydantic import BaseModel, ConfigDict, Field

from atagia.engine import Atagia
from benchmarks.atagia_bench_gate.dataset import (
    GateProbeConversation,
    GateProbePersona,
    GateProbeQuestion,
    GateSuiteDataset,
    load_gate_suite,
)
from benchmarks.atagia_bench_gate.scoring import (
    GateObservation,
    GateQuestionScore,
    GateSuiteScore,
    aggregate_scores,
    extract_gate_observation,
    score_question,
)
from benchmarks.llm_config import provider_api_key_kwargs


_ADAPTIVE_RETRIEVAL_ENV = "ATAGIA_ADAPTIVE_RETRIEVAL"

# The gate suite runs with privacy enforcement off, consistent with the
# project's benchmark-validation policy: the memory engine is proven first,
# privacy gates are a separate later workstream. The user_id partition stays
# mandatory regardless.
_PRIVACY_ENFORCEMENT = "off"


class GateRunQuestionResult(BaseModel):
    """End-to-end result for one probe question."""

    model_config = ConfigDict(extra="forbid")

    persona_id: str
    score: GateQuestionScore
    response_text: str
    retrieval_time_ms: float = Field(ge=0.0)


class GateRunReport(BaseModel):
    """Full report for one gate-suite run."""

    model_config = ConfigDict(extra="forbid")

    suite_name: str
    adaptive_retrieval_enabled: bool
    results: list[GateRunQuestionResult] = Field(default_factory=list)
    score: GateSuiteScore
    model_info: dict[str, Any] = Field(default_factory=dict)


@contextmanager
def _adaptive_retrieval_env(enabled: bool) -> Iterator[None]:
    """Temporarily set the adaptive-retrieval env lever for the engine default.

    Restores the previous value on exit so the runner does not leak global
    state into other benchmarks sharing the same process.
    """
    previous = os.environ.get(_ADAPTIVE_RETRIEVAL_ENV)
    os.environ[_ADAPTIVE_RETRIEVAL_ENV] = "1" if enabled else "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_ADAPTIVE_RETRIEVAL_ENV, None)
        else:
            os.environ[_ADAPTIVE_RETRIEVAL_ENV] = previous


class GateSuiteRunner:
    """Run the adaptive retrieval gate suite end to end with the flag ON."""

    def __init__(
        self,
        *,
        dataset: GateSuiteDataset | None = None,
        manifests_dir: str | Path | None = None,
        llm_provider: str | None = None,
        llm_api_key: str | None = None,
        forced_global_model: str | None = None,
        chat_model: str | None = None,
        ingest_model: str | None = None,
        retrieval_model: str | None = None,
        adaptive_retrieval: bool = True,
    ) -> None:
        self._dataset = dataset if dataset is not None else load_gate_suite()
        self._manifests_dir = str(manifests_dir) if manifests_dir else None
        self._llm_provider = llm_provider
        self._llm_api_key = llm_api_key
        self._forced_global_model = forced_global_model
        self._chat_model = chat_model
        self._ingest_model = ingest_model
        self._retrieval_model = retrieval_model
        self._adaptive_retrieval = adaptive_retrieval

    def _model_kwargs(self) -> dict[str, Any]:
        """Return Atagia constructor kwargs for model routing."""
        if self._forced_global_model is not None:
            return {"llm_forced_global_model": self._forced_global_model}
        kwargs: dict[str, Any] = {}
        if self._chat_model is not None:
            kwargs["llm_chat_model"] = self._chat_model
        if self._ingest_model is not None:
            kwargs["llm_ingest_model"] = self._ingest_model
        if self._retrieval_model is not None:
            kwargs["llm_retrieval_model"] = self._retrieval_model
        return kwargs

    async def run(self, db_path: str | Path) -> GateRunReport:
        """Ingest every persona and score every probe question."""
        results: list[GateRunQuestionResult] = []

        with _adaptive_retrieval_env(self._adaptive_retrieval):
            for persona in self._dataset.personas:
                persona_results = await self._run_persona(
                    persona,
                    db_path=Path(db_path),
                )
                results.extend(persona_results)

        suite_score = aggregate_scores([result.score for result in results])
        return GateRunReport(
            suite_name=self._dataset.name,
            adaptive_retrieval_enabled=self._adaptive_retrieval,
            results=results,
            score=suite_score,
            model_info={
                "provider": self._llm_provider,
                "forced_global_model": self._forced_global_model,
                "chat_model": self._chat_model,
                "ingest_model": self._ingest_model,
                "retrieval_model": self._retrieval_model,
            },
        )

    async def _run_persona(
        self,
        persona: GateProbePersona,
        *,
        db_path: Path,
    ) -> list[GateRunQuestionResult]:
        """Ingest one persona's conversations and answer its probe questions."""
        persona_db = db_path.parent / f"{db_path.stem}_{persona.persona_id}.db"
        results: list[GateRunQuestionResult] = []

        async with Atagia(
            db_path=persona_db,
            manifests_dir=self._manifests_dir,
            **self._model_kwargs(),
            **provider_api_key_kwargs(self._llm_provider, self._llm_api_key),
        ) as engine:
            runtime = engine.runtime
            if runtime is None:
                raise RuntimeError("Atagia runtime unavailable")

            user_id = f"gate-{persona.persona_id}"
            await engine.create_user(user_id)

            for conversation in persona.conversations:
                await self._ingest_conversation(
                    engine,
                    user_id=user_id,
                    conversation=conversation,
                )

            for question in persona.questions:
                results.append(
                    await self._run_question(
                        engine,
                        user_id=user_id,
                        persona_id=persona.persona_id,
                        question=question,
                    )
                )

        return results

    async def _ingest_conversation(
        self,
        engine: Atagia,
        *,
        user_id: str,
        conversation: GateProbeConversation,
    ) -> None:
        """Ingest one setup conversation and drain workers."""
        await engine.create_conversation(
            user_id,
            conversation.conversation_id,
            assistant_mode_id=conversation.assistant_mode_id,
        )
        for turn in conversation.turns:
            await engine.ingest_message(
                user_id=user_id,
                conversation_id=conversation.conversation_id,
                role=turn.role,
                text=turn.text,
                occurred_at=turn.timestamp or None,
                privacy_enforcement=_PRIVACY_ENFORCEMENT,
                authenticated_user_privilege_level="atagia_master",
                authenticated_user_is_atagia_master=True,
            )
        drained = await engine.flush(timeout_seconds=1800.0)
        if not drained:
            raise RuntimeError(
                f"Timed out draining workers for conversation "
                f"{conversation.conversation_id}"
            )

    async def _run_question(
        self,
        engine: Atagia,
        *,
        user_id: str,
        persona_id: str,
        question: GateProbeQuestion,
    ) -> GateRunQuestionResult:
        """Ask one probe question with the gate enabled and score the result."""
        chat_kwargs: dict[str, Any] = {
            "user_id": user_id,
            "conversation_id": question.target_conversation_id,
            "message": question.question_text,
            "mode": question.assistant_mode_id,
            "debug": True,
            "privacy_enforcement": _PRIVACY_ENFORCEMENT,
            "authenticated_user_privilege_level": "atagia_master",
            "authenticated_user_is_atagia_master": True,
            "adaptive_retrieval": self._adaptive_retrieval,
        }

        started = perf_counter()
        chat_result = await engine.chat(**chat_kwargs)
        retrieval_time_ms = (perf_counter() - started) * 1000.0

        observation = self._observation_from_chat_result(chat_result)
        score = score_question(question, observation)
        return GateRunQuestionResult(
            persona_id=persona_id,
            score=score,
            response_text=chat_result.response_text,
            retrieval_time_ms=retrieval_time_ms,
        )

    @staticmethod
    def _observation_from_chat_result(chat_result: Any) -> GateObservation:
        """Extract a gate observation from a chat result's debug payload."""
        debug = chat_result.debug if isinstance(chat_result.debug, dict) else None
        return extract_gate_observation(debug)
