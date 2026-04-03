"""API tests for admin metrics routes."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from atagia.app import create_app
from atagia.core.clock import FrozenClock
from atagia.core.config import Settings
from atagia.core.metrics_repository import MetricsRepository
from atagia.core.repositories import ConversationRepository, MemoryObjectRepository, MessageRepository, UserRepository
from atagia.core.retrieval_event_repository import (
    AdminAuditRepository,
    MemoryFeedbackRepository,
    RetrievalEventRepository,
)
from atagia.models.schemas_memory import MemoryObjectType, MemoryScope, MemorySourceKind

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "manifests"


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        sqlite_path=str(tmp_path / "atagia-admin-metrics.db"),
        migrations_path=str(MIGRATIONS_DIR),
        manifests_path=str(MANIFESTS_DIR),
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        llm_provider="openai",
        llm_api_key=None,
        openai_api_key="test-openai-key",
        openrouter_api_key=None,
        llm_base_url=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
        llm_extraction_model="chat-test-model",
        llm_scoring_model="score-test-model",
        llm_classifier_model="classify-test-model",
        llm_chat_model="reply-test-model",
        service_mode=True,
        service_api_key="service-key",
        admin_api_key="admin-key",
        workers_enabled=False,
        debug=False,
    )


@contextmanager
def _connection(client: TestClient):
    connection = client.portal.call(client.app.state.runtime.open_connection)
    try:
        yield connection
    finally:
        client.portal.call(connection.close)


def test_admin_metrics_routes_require_admin_key(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        assert client.get("/v1/admin/metrics/latest").status_code == 401
        assert client.get("/v1/admin/metrics/mur/history").status_code == 401
        assert client.post(
            "/v1/admin/metrics/compute",
            json={"time_bucket": "2026-03-31", "metrics": ["mur"]},
        ).status_code == 401
        assert client.get(
            "/v1/admin/metrics/retrieval-summary",
            params={"from_date": "2026-03-31", "to_date": "2026-03-31"},
        ).status_code == 401


def test_admin_metrics_validate_metric_names_at_api_boundary(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        history = client.get(
            "/v1/admin/metrics/not-a-metric/history",
            headers={"Authorization": "Bearer admin-key"},
        )
        compute = client.post(
            "/v1/admin/metrics/compute",
            json={"time_bucket": "2026-03-31", "metrics": ["not-a-metric"]},
            headers={"Authorization": "Bearer admin-key"},
        )

        assert history.status_code == 422
        assert compute.status_code == 422


def test_admin_metrics_routes_support_compute_latest_history_and_summary(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        runtime = client.app.state.runtime
        runtime.clock = FrozenClock(datetime(2026, 3, 31, 9, 0, tzinfo=timezone.utc))
        with _connection(client) as connection:
            users = UserRepository(connection, runtime.clock)
            conversations = ConversationRepository(connection, runtime.clock)
            messages = MessageRepository(connection, runtime.clock)
            memories = MemoryObjectRepository(connection, runtime.clock)
            events = RetrievalEventRepository(connection, runtime.clock)
            feedback = MemoryFeedbackRepository(connection, runtime.clock)
            metrics = MetricsRepository(connection, runtime.clock)

            client.portal.call(users.create_user, "usr_1")
            client.portal.call(
                conversations.create_conversation,
                "cnv_1",
                "usr_1",
                None,
                "coding_debug",
                "Chat",
            )
            client.portal.call(messages.create_message, "msg_1", "cnv_1", "user", 1, "Need help", 2, {})
            client.portal.call(messages.create_message, "msg_2", "cnv_1", "assistant", 2, "Try this", 2, {})
            client.portal.call(
                lambda: memories.create_memory_object(
                    user_id="usr_1",
                    conversation_id="cnv_1",
                    assistant_mode_id="coding_debug",
                    object_type=MemoryObjectType.EVIDENCE,
                    scope=MemoryScope.CONVERSATION,
                    canonical_text="Retry memory",
                    source_kind=MemorySourceKind.EXTRACTED,
                    confidence=0.9,
                    privacy_level=0,
                    memory_id="mem_1",
                )
            )
            client.portal.call(
                events.create_event,
                {
                    "id": "ret_1",
                    "user_id": "usr_1",
                    "conversation_id": "cnv_1",
                    "request_message_id": "msg_1",
                    "response_message_id": "msg_2",
                    "assistant_mode_id": "coding_debug",
                    "retrieval_plan_json": {"fts_queries": ["retry"]},
                    "selected_memory_ids_json": ["mem_1"],
                    "context_view_json": {
                        "selected_memory_ids": ["mem_1"],
                        "items_included": 1,
                        "items_dropped": 0,
                        "total_tokens_estimate": 80,
                    },
                    "outcome_json": {"cold_start": False, "zero_candidates": False},
                    "created_at": "2026-03-31T09:00:03+00:00",
                },
            )
            client.portal.call(
                lambda: feedback.create_feedback(
                    retrieval_event_id="ret_1",
                    memory_id="mem_1",
                    user_id="usr_1",
                    feedback_type="useful",
                    score=1.0,
                    metadata={},
                )
            )
            client.portal.call(
                lambda: metrics.store_metric(
                    metric_name="mur",
                    value=0.25,
                    sample_count=4,
                    user_id="usr_1",
                    assistant_mode_id="coding_debug",
                    time_bucket="2026-03-30",
                    computed_at="2026-03-30T12:00:00+00:00",
                )
            )

            compute_response = client.post(
                "/v1/admin/metrics/compute",
                json={
                    "time_bucket": "2026-03-31",
                    "user_id": "usr_1",
                    "assistant_mode_id": "coding_debug",
                    "metrics": ["mur", "ccr"],
                },
                headers={"Authorization": "Bearer admin-key"},
            )
            assert compute_response.status_code == 200
            compute_payload = compute_response.json()
            assert compute_payload["computed"]["mur"]["value"] == 1.0
            assert compute_payload["queued_metrics"] == ["ccr"]

            queued = client.portal.call(runtime.storage_backend.dequeue_job, "stream:atagia:evaluate", 0)
            assert queued is not None
            assert queued["payload"]["job_type"] == "run_evaluation"

            latest_response = client.get(
                "/v1/admin/metrics/latest",
                params={"user_id": "usr_1", "assistant_mode_id": "coding_debug"},
                headers={"Authorization": "Bearer admin-key"},
            )
            assert latest_response.status_code == 200
            assert latest_response.json()["mur"]["metric_value"] == 1.0

            history_response = client.get(
                "/v1/admin/metrics/mur/history",
                params={"user_id": "usr_1", "assistant_mode_id": "coding_debug", "limit": 5},
                headers={"Authorization": "Bearer admin-key"},
            )
            assert history_response.status_code == 200
            assert [row["time_bucket"] for row in history_response.json()] == ["2026-03-31", "2026-03-30"]

            summary_response = client.get(
                "/v1/admin/metrics/retrieval-summary",
                params={"user_id": "usr_1", "from_date": "2026-03-31", "to_date": "2026-03-31"},
                headers={"Authorization": "Bearer admin-key"},
            )
            assert summary_response.status_code == 200
            assert summary_response.json()["total_events"] == 1
            assert summary_response.json()["avg_items_included"] == 1.0

            audit_entries = client.portal.call(AdminAuditRepository(connection, runtime.clock).list_entries)
            audit_actions = [entry["action"] for entry in audit_entries]
            assert "metrics_compute" in audit_actions
            assert "metrics_latest" in audit_actions
            assert "metrics_history" in audit_actions
            assert "metrics_retrieval_summary" in audit_actions
