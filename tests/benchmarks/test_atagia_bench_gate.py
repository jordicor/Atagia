"""Tests for the adaptive retrieval gate diagnostic suite (pure-python parts)."""

from __future__ import annotations

import pytest

from atagia.models.schemas_memory import AdaptiveGateStatus, MemoryDependence
from benchmarks.atagia_bench_gate.dataset import (
    GateProbeQuestion,
    GateSuiteDataset,
    load_gate_suite,
)
from benchmarks.atagia_bench_gate.scoring import (
    GateObservation,
    aggregate_scores,
    extract_gate_observation,
    score_question,
)


# ---- Dataset loading and validation ----


def test_bundled_dataset_loads_and_validates() -> None:
    dataset = load_gate_suite()
    assert dataset.name == "atagia-bench-gate-v0"
    assert 20 <= dataset.total_questions <= 24
    assert dataset.total_conversations == len(dataset.personas)


def test_bundled_dataset_has_world_and_personal_pairs() -> None:
    dataset = load_gate_suite()
    pair_members: dict[str, list[MemoryDependence]] = {}
    for question in dataset.questions:
        if question.pair_id is not None:
            pair_members.setdefault(question.pair_id, []).append(
                question.expected_memory_dependence
            )
    assert pair_members, "Expected at least one paired topic"
    for members in pair_members.values():
        assert sorted(m.value for m in members) == ["personal", "world"]


def test_bundled_dataset_has_conversation_window_probes() -> None:
    dataset = load_gate_suite()
    conversation_probes = [
        question
        for question in dataset.questions
        if question.expected_memory_dependence is MemoryDependence.CONVERSATION
    ]
    assert conversation_probes, "Expected at least one conversation-window probe"
    assert all(q.probe_kind == "conversation" for q in conversation_probes)


def test_bundled_dataset_has_at_least_two_non_english_pairs() -> None:
    dataset = load_gate_suite()
    non_english_pairs: set[str] = set()
    for question in dataset.questions:
        if question.pair_id is not None and question.language != "en":
            non_english_pairs.add(question.pair_id)
    assert len(non_english_pairs) >= 2


def test_expected_action_skips_world_and_conversation() -> None:
    for dependence, expected_skip in (
        (MemoryDependence.WORLD, True),
        (MemoryDependence.CONVERSATION, True),
        (MemoryDependence.PERSONAL, False),
        (MemoryDependence.MIXED, False),
    ):
        question = GateProbeQuestion(
            question_id=f"q-{dependence.value}",
            question_text="probe",
            probe_kind="single",
            target_conversation_id="conv",
            assistant_mode_id="personal_assistant",
            expected_memory_dependence=dependence,
        )
        assert question.expected_action.skip_retrieval is expected_skip


def test_unknown_conversation_target_is_rejected() -> None:
    payload = {
        "name": "bad",
        "personas": [
            {
                "persona_id": "p1",
                "display_name": "P One",
                "profile": "x",
                "conversations": [
                    {
                        "conversation_id": "conv-a",
                        "assistant_mode_id": "personal_assistant",
                        "timestamp_base": "2025-01-01T00:00:00",
                        "turns": [],
                    }
                ],
                "questions": [
                    {
                        "question_id": "q1",
                        "question_text": "probe",
                        "probe_kind": "single",
                        "target_conversation_id": "conv-missing",
                        "assistant_mode_id": "personal_assistant",
                        "expected_memory_dependence": "world",
                    }
                ],
            }
        ],
    }
    with pytest.raises(ValueError, match="unknown conversation"):
        GateSuiteDataset.model_validate(payload)


def test_malformed_pair_is_rejected() -> None:
    payload = {
        "name": "bad",
        "personas": [
            {
                "persona_id": "p1",
                "display_name": "P One",
                "profile": "x",
                "conversations": [
                    {
                        "conversation_id": "conv-a",
                        "assistant_mode_id": "personal_assistant",
                        "timestamp_base": "2025-01-01T00:00:00",
                        "turns": [],
                    }
                ],
                "questions": [
                    {
                        "question_id": "q1",
                        "question_text": "probe a",
                        "probe_kind": "paired",
                        "pair_id": "topic",
                        "target_conversation_id": "conv-a",
                        "assistant_mode_id": "personal_assistant",
                        "expected_memory_dependence": "world",
                    },
                    {
                        "question_id": "q2",
                        "question_text": "probe b",
                        "probe_kind": "paired",
                        "pair_id": "topic",
                        "target_conversation_id": "conv-a",
                        "assistant_mode_id": "personal_assistant",
                        "expected_memory_dependence": "world",
                    },
                ],
            }
        ],
    }
    with pytest.raises(ValueError, match="exactly one 'world' and one 'personal'"):
        GateSuiteDataset.model_validate(payload)


def test_duplicate_question_id_is_rejected() -> None:
    payload = {
        "name": "bad",
        "personas": [
            {
                "persona_id": "p1",
                "display_name": "P One",
                "profile": "x",
                "conversations": [
                    {
                        "conversation_id": "conv-a",
                        "assistant_mode_id": "personal_assistant",
                        "timestamp_base": "2025-01-01T00:00:00",
                        "turns": [],
                    }
                ],
                "questions": [
                    {
                        "question_id": "dup",
                        "question_text": "probe a",
                        "probe_kind": "single",
                        "target_conversation_id": "conv-a",
                        "assistant_mode_id": "personal_assistant",
                        "expected_memory_dependence": "world",
                    },
                    {
                        "question_id": "dup",
                        "question_text": "probe b",
                        "probe_kind": "single",
                        "target_conversation_id": "conv-a",
                        "assistant_mode_id": "personal_assistant",
                        "expected_memory_dependence": "personal",
                    },
                ],
            }
        ],
    }
    with pytest.raises(ValueError, match="Duplicate question id"):
        GateSuiteDataset.model_validate(payload)


# ---- Gate observation extraction ----


def test_extract_classification_from_need_detection_trace() -> None:
    debug = {
        "retrieval_trace": {
            "need_detection": {"memory_dependence": "world"},
        }
    }
    observation = extract_gate_observation(debug)
    assert observation.classification is MemoryDependence.WORLD
    assert observation.gate_status is None
    assert observation.retrieval_skipped is None


def test_extract_status_and_classification_from_plan_object() -> None:
    debug = {
        "retrieval_plan": {
            "adaptive_gate": {"status": "skipped", "classification": "world"},
        },
        "selected_memory_ids": [],
    }
    observation = extract_gate_observation(debug)
    assert observation.gate_status is AdaptiveGateStatus.SKIPPED
    assert observation.classification is MemoryDependence.WORLD
    assert observation.retrieval_skipped is True
    assert observation.selected_memory_count == 0


def test_extract_status_from_scalar_string_entry() -> None:
    debug = {
        "retrieval_plan": {"adaptive_gate": "not_applicable"},
    }
    observation = extract_gate_observation(debug)
    assert observation.gate_status is AdaptiveGateStatus.NOT_APPLICABLE
    assert observation.retrieval_skipped is None


def test_extract_retrieved_status_means_not_skipped() -> None:
    debug = {
        "retrieval_diagnostics_for_guard": {
            "adaptive_gate": {"status": "retrieved", "classification": "personal"},
            "selected_memory_count": 3,
        },
    }
    observation = extract_gate_observation(debug)
    assert observation.gate_status is AdaptiveGateStatus.RETRIEVED
    assert observation.retrieval_skipped is False
    assert observation.classification is MemoryDependence.PERSONAL
    assert observation.selected_memory_count == 3


def test_extract_prefers_plan_over_diagnostics_on_conflict() -> None:
    # The scorer reads ``retrieval_plan`` before ``retrieval_diagnostics_for_guard``
    # and keeps the first non-None value, so when both surfaces disagree the plan
    # block wins for both the status and the classification.
    debug = {
        "retrieval_plan": {
            "adaptive_gate": {"status": "skipped", "classification": "world"},
        },
        "retrieval_diagnostics_for_guard": {
            "adaptive_gate": {"status": "retrieved", "classification": "personal"},
            "selected_memory_count": 3,
        },
    }
    observation = extract_gate_observation(debug)
    assert observation.gate_status is AdaptiveGateStatus.SKIPPED
    assert observation.classification is MemoryDependence.WORLD
    assert observation.retrieval_skipped is True


def test_extract_handles_missing_debug() -> None:
    observation = extract_gate_observation(None)
    assert observation.classification is None
    assert observation.gate_status is None
    assert observation.retrieval_skipped is None


def test_extract_ignores_unparseable_values() -> None:
    debug = {
        "retrieval_plan": {"adaptive_gate": {"status": "??", "classification": "??"}},
        "retrieval_trace": {"need_detection": {"memory_dependence": "??"}},
    }
    observation = extract_gate_observation(debug)
    assert observation.classification is None
    assert observation.gate_status is None


# ---- Scoring ----


def _world_question() -> GateProbeQuestion:
    return GateProbeQuestion(
        question_id="q-world",
        question_text="probe",
        probe_kind="paired",
        pair_id="topic",
        target_conversation_id="conv",
        assistant_mode_id="personal_assistant",
        expected_memory_dependence=MemoryDependence.WORLD,
    )


def _personal_question() -> GateProbeQuestion:
    return GateProbeQuestion(
        question_id="q-personal",
        question_text="probe",
        probe_kind="paired",
        pair_id="topic",
        target_conversation_id="conv",
        assistant_mode_id="personal_assistant",
        expected_memory_dependence=MemoryDependence.PERSONAL,
    )


def test_score_question_correct_classification_and_action() -> None:
    observation = GateObservation(
        classification=MemoryDependence.WORLD,
        gate_status=AdaptiveGateStatus.SKIPPED,
        retrieval_skipped=True,
    )
    score = score_question(_world_question(), observation)
    assert score.classification_correct is True
    assert score.action_correct is True
    assert score.expected_skip is True


def test_score_question_classification_unscored_when_absent() -> None:
    observation = GateObservation()
    score = score_question(_world_question(), observation)
    assert score.classification_correct is None
    assert score.action_correct is None


def test_aggregate_counts_false_skip() -> None:
    # Personal turn was wrongly skipped: a false skip (dangerous).
    observation = GateObservation(
        classification=MemoryDependence.WORLD,
        gate_status=AdaptiveGateStatus.SKIPPED,
        retrieval_skipped=True,
    )
    score = score_question(_personal_question(), observation)
    suite = aggregate_scores([score])
    assert suite.false_skips == 1
    assert suite.correct_skips == 0
    assert suite.missed_skips == 0
    assert suite.classification_correct == 0
    assert suite.classification_scored == 1
    assert suite.action_correct == 0
    assert suite.action_scored == 1


def test_aggregate_counts_correct_skip_and_missed_skip() -> None:
    correct_skip = score_question(
        _world_question(),
        GateObservation(
            classification=MemoryDependence.WORLD,
            gate_status=AdaptiveGateStatus.SKIPPED,
            retrieval_skipped=True,
        ),
    )
    missed_skip = score_question(
        _world_question(),
        GateObservation(
            classification=MemoryDependence.WORLD,
            gate_status=AdaptiveGateStatus.RETRIEVED,
            retrieval_skipped=False,
        ),
    )
    suite = aggregate_scores([correct_skip, missed_skip])
    assert suite.correct_skips == 1
    assert suite.missed_skips == 1
    assert suite.false_skips == 0
    # The missed skip retrieved a world turn, so its action is wrong; the
    # correct skip's action is right.
    assert suite.action_correct == 1
    assert suite.action_scored == 2


def test_aggregate_classification_accuracy_by_language() -> None:
    en_question = GateProbeQuestion(
        question_id="q-en",
        question_text="probe",
        language="en",
        probe_kind="single",
        target_conversation_id="conv",
        assistant_mode_id="personal_assistant",
        expected_memory_dependence=MemoryDependence.WORLD,
    )
    es_question = GateProbeQuestion(
        question_id="q-es",
        question_text="probe",
        language="es",
        probe_kind="single",
        target_conversation_id="conv",
        assistant_mode_id="personal_assistant",
        expected_memory_dependence=MemoryDependence.PERSONAL,
    )
    scores = [
        score_question(
            en_question,
            GateObservation(classification=MemoryDependence.WORLD),
        ),
        score_question(
            es_question,
            GateObservation(classification=MemoryDependence.WORLD),
        ),
    ]
    suite = aggregate_scores(scores)
    assert suite.classification_accuracy == pytest.approx(0.5)
    assert suite.classification_accuracy_by_language["en"] == pytest.approx(1.0)
    assert suite.classification_accuracy_by_language["es"] == pytest.approx(0.0)


def test_aggregate_empty_accuracy_is_none() -> None:
    suite = aggregate_scores([])
    assert suite.classification_accuracy is None
    assert suite.action_accuracy is None
