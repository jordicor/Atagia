"""Tests for operational profile snapshots on worker envelopes."""

from __future__ import annotations

from pathlib import Path

from atagia.memory.operational_profile import OperationalProfileLoader
from atagia.models.schemas_jobs import JobEnvelope, JobType

PROFILES_DIR = Path(__file__).resolve().parents[2] / "operational_profiles"


def test_job_envelope_round_trips_operational_snapshot() -> None:
    loader = OperationalProfileLoader(PROFILES_DIR)
    loader.load_all()
    snapshot = loader.resolve(operational_profile="offline").snapshot

    envelope = JobEnvelope(
        job_id="job_1",
        job_type=JobType.EXTRACT_MEMORY_CANDIDATES,
        user_id="usr_1",
        conversation_id="cnv_1",
        message_ids=["msg_1"],
        payload={"message_id": "msg_1"},
        operational_profile=snapshot,
    )
    round_tripped = JobEnvelope.model_validate(envelope.model_dump(mode="json"))

    assert round_tripped.operational_profile is not None
    assert round_tripped.operational_profile.profile_id == "offline"
    assert round_tripped.operational_profile.token == snapshot.token


def test_job_envelope_without_operational_snapshot_remains_valid() -> None:
    envelope = JobEnvelope(
        job_id="job_1",
        job_type=JobType.EXTRACT_MEMORY_CANDIDATES,
        user_id="usr_1",
    )

    assert envelope.operational_profile is None
