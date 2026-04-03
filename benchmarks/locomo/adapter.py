"""Adapter for the LoCoMo benchmark dataset."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any

from benchmarks.base import BenchmarkAdapter, BenchmarkConversation, BenchmarkDataset, BenchmarkQuestion, BenchmarkTurn

_SESSION_KEY_PATTERN = re.compile(r"^session_(\d+)$")


class LoCoMoAdapter(BenchmarkAdapter):
    """Load and normalize LoCoMo conversations from `locomo10.json`."""

    def __init__(self, data_path: str | Path) -> None:
        self._data_path = Path(data_path).expanduser()

    def load(self) -> BenchmarkDataset:
        """Read LoCoMo JSON and return normalized benchmark models."""
        raw_samples = json.loads(self._data_path.read_text(encoding="utf-8"))
        conversations = [
            self._parse_sample(sample, index)
            for index, sample in enumerate(raw_samples, start=1)
        ]
        return BenchmarkDataset(name="LoCoMo", conversations=conversations)

    def _parse_sample(self, sample: dict[str, Any], index: int) -> BenchmarkConversation:
        conversation_id = str(sample.get("sample_id") or f"locomo-{index:02d}")
        conversation_payload = sample.get("conversation")
        if not isinstance(conversation_payload, dict):
            raise ValueError(f"LoCoMo sample {conversation_id} is missing a conversation payload")

        turns = self._parse_turns(conversation_payload)
        questions = self._parse_questions(sample.get("qa"), conversation_id)
        return BenchmarkConversation(
            conversation_id=conversation_id,
            turns=turns,
            questions=questions,
        )

    def _parse_turns(self, conversation_payload: dict[str, Any]) -> list[BenchmarkTurn]:
        session_keys = self._ordered_session_keys(conversation_payload)
        speaker_roles = self._speaker_role_map(conversation_payload, session_keys)
        turns: list[BenchmarkTurn] = []
        for session_key in session_keys:
            session_timestamp = self._normalize_timestamp(
                conversation_payload.get(f"{session_key}_date_time")
            )
            session_turns = conversation_payload.get(session_key)
            if not isinstance(session_turns, list):
                continue
            for turn in session_turns:
                if not isinstance(turn, dict):
                    continue
                speaker = str(turn.get("speaker", ""))
                turns.append(
                    BenchmarkTurn(
                        role=speaker_roles.get(speaker, "user"),
                        text=str(turn.get("text", "")),
                        speaker=speaker,
                        session_id=session_key,
                        timestamp=session_timestamp,
                        turn_id=(
                            str(turn.get("dia_id"))
                            if turn.get("dia_id") is not None
                            else None
                        ),
                    )
                )
        return turns

    def _parse_questions(
        self,
        raw_questions: Any,
        conversation_id: str,
    ) -> list[BenchmarkQuestion]:
        if not isinstance(raw_questions, list):
            raise ValueError(f"LoCoMo sample {conversation_id} is missing its question list")
        questions: list[BenchmarkQuestion] = []
        for index, item in enumerate(raw_questions, start=1):
            if not isinstance(item, dict):
                continue
            evidence = item.get("evidence")
            questions.append(
                BenchmarkQuestion(
                    question_text=str(item.get("question", "")),
                    ground_truth=self._stringify_value(item.get("answer")),
                    category=int(item["category"]),
                    evidence_turn_ids=[
                        str(entry)
                        for entry in evidence
                    ] if isinstance(evidence, list) else [],
                    question_id=f"{conversation_id}:q{index}",
                )
            )
        return questions

    @staticmethod
    def _ordered_session_keys(conversation_payload: dict[str, Any]) -> list[str]:
        ordered = [
            (int(match.group(1)), key)
            for key in conversation_payload
            if (match := _SESSION_KEY_PATTERN.match(key)) is not None
        ]
        ordered.sort(key=lambda item: item[0])
        return [key for _, key in ordered]

    @staticmethod
    def _speaker_role_map(
        conversation_payload: dict[str, Any],
        session_keys: list[str],
    ) -> dict[str, str]:
        ordered_speakers: list[str] = []
        for session_key in session_keys:
            session_turns = conversation_payload.get(session_key)
            if not isinstance(session_turns, list):
                continue
            for turn in session_turns:
                if not isinstance(turn, dict):
                    continue
                speaker = turn.get("speaker")
                if not isinstance(speaker, str):
                    continue
                if speaker not in ordered_speakers:
                    ordered_speakers.append(speaker)
                if len(ordered_speakers) >= 2:
                    break
            if len(ordered_speakers) >= 2:
                break

        if len(ordered_speakers) < 2:
            for fallback_key in ("speaker_a", "speaker_b"):
                speaker = conversation_payload.get(fallback_key)
                if isinstance(speaker, str) and speaker not in ordered_speakers:
                    ordered_speakers.append(speaker)

        if not ordered_speakers:
            return {}
        if len(ordered_speakers) == 1:
            return {ordered_speakers[0]: "user"}
        return {
            ordered_speakers[0]: "user",
            ordered_speakers[1]: "assistant",
        }

    @staticmethod
    def _normalize_timestamp(raw_value: Any) -> str:
        if not isinstance(raw_value, str) or not raw_value.strip():
            return ""
        normalized_value = re.sub(r"\b(am|pm)\b", lambda match: match.group(1).upper(), raw_value.strip())
        for fmt in (
            "%I:%M %p on %d %B, %Y",
            "%I:%M %p on %B %d, %Y",
        ):
            try:
                return datetime.strptime(normalized_value, fmt).isoformat()
            except ValueError:
                continue
        return raw_value

    @staticmethod
    def _stringify_value(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)
