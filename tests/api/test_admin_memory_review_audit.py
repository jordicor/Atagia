"""Audit contract for the sensitive admin memory-review read."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Iterator

from fastapi import HTTPException
from fastapi.testclient import TestClient
import pytest

from atagia.api import routes_admin
from atagia.app import create_app
from atagia.core.config import Settings
from atagia.core.repositories import (
    ConversationRepository,
    MemoryObjectRepository,
    UserRepository,
)
from atagia.core.retrieval_event_repository import AdminAuditRepository
from atagia.models.schemas_memory import (
    MemoryCategory,
    MemoryObjectType,
    MemoryScope,
    MemorySourceKind,
    MemoryStatus,
)
from atagia.services.errors import (
    TranscriptRebuildInProgressError,
    TranscriptRebuildRemediationRequiredError,
)

MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "migrations"
)
MANIFESTS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "atagia" / "resources" / "manifests"
)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        sqlite_path=str(tmp_path / "memory-review-audit.db"),
        migrations_path=str(MIGRATIONS_DIR),
        manifests_path=str(MANIFESTS_DIR),
        storage_backend="inprocess",
        redis_url="redis://localhost:6379/0",
        openai_api_key="test-openai-key",
        openrouter_api_key=None,
        openrouter_site_url="http://localhost",
        openrouter_app_name="Atagia",
        llm_chat_model="openai/test-model",
        llm_forced_global_model="openai/test-model",
        service_mode=True,
        service_api_key="service-key",
        admin_api_key="admin-key-secret",
        workers_enabled=False,
        debug=False,
        allow_insecure_http=True,
    )


@contextmanager
def _connection(client: TestClient) -> Iterator[Any]:
    connection = client.portal.call(client.app.state.runtime.open_connection)
    try:
        yield connection
    finally:
        client.portal.call(connection.close)


def _seed_sensitive_memory(client: TestClient) -> None:
    with _connection(client) as connection:
        runtime = client.app.state.runtime
        users = UserRepository(connection, runtime.clock)
        conversations = ConversationRepository(connection, runtime.clock)
        memories = MemoryObjectRepository(connection, runtime.clock)
        client.portal.call(users.create_user, "usr_audit")
        client.portal.call(
            lambda: conversations.create_conversation(
                "cnv_audit",
                "usr_audit",
                None,
                "personal_assistant",
                "Sensitive review",
                platform_id="audit-platform",
            )
        )
        client.portal.call(
            lambda: memories.create_memory_object(
                memory_id="mem_audit",
                user_id="usr_audit",
                conversation_id="cnv_audit",
                assistant_mode_id="personal_assistant",
                object_type=MemoryObjectType.EVIDENCE,
                scope=MemoryScope.USER,
                canonical_text="Imported bank PIN: 4512",
                index_text="bank PIN",
                payload={
                    "ingest_origin": "backfill",
                    "raw_source": "private source payload 4512",
                    "source_message_ids": ["private-source-message-4512"],
                },
                source_kind=MemorySourceKind.EXTRACTED,
                confidence=0.97,
                privacy_level=3,
                memory_category=MemoryCategory.PIN_OR_PASSWORD,
                preserve_verbatim=True,
                status=MemoryStatus.REVIEW_REQUIRED,
                platform_id="audit-platform",
            )
        )


def _audit_entries(client: TestClient) -> list[dict[str, Any]]:
    with _connection(client) as connection:
        runtime = client.app.state.runtime
        return client.portal.call(
            AdminAuditRepository(connection, runtime.clock).list_entries
        )


def test_successful_memory_review_read_is_audited_without_sensitive_content(
    tmp_path: Path,
) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        _seed_sensitive_memory(client)

        response = client.get(
            "/v1/admin/memory-review",
            headers={"Authorization": "Bearer admin-key-secret"},
            params={"user_id": "usr_audit", "platform_id": "audit-platform"},
        )
        entries = _audit_entries(client)

    assert response.status_code == 200
    assert response.json()["items"][0]["canonical_text"] == "Imported bank PIN: 4512"
    assert len(entries) == 1
    entry = entries[0]
    assert entry["admin_user_id"] == "admin_api_key"
    assert entry["action"] == "list_review_required_memories"
    assert entry["target_id"] == "usr_audit"
    assert entry["metadata_json"]["status"] == "success"
    assert entry["metadata_json"]["result_count"] == 1
    encoded_audit = json.dumps(entry, ensure_ascii=False)
    for forbidden in (
        "Imported bank PIN: 4512",
        "private source payload 4512",
        "private-source-message-4512",
        "admin-key-secret",
    ):
        assert forbidden not in encoded_audit


def test_denied_memory_review_attempt_is_attributable_and_does_not_log_key(
    tmp_path: Path,
) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        response = client.get(
            "/v1/admin/memory-review",
            headers={"Authorization": "Bearer wrong-secret"},
            params={"platform_id": "audit-platform"},
        )
        entries = _audit_entries(client)

    assert response.status_code == 401
    assert len(entries) == 1
    assert entries[0]["admin_user_id"] == "unauthenticated_admin_request"
    assert entries[0]["metadata_json"]["status"] == "denied"
    assert entries[0]["metadata_json"]["http_status"] == 401
    assert "wrong-secret" not in json.dumps(entries[0])


def test_memory_review_rejects_duplicate_authorization_before_authentication(
    tmp_path: Path,
) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        response = client.get(
            "/v1/admin/memory-review",
            headers=[
                ("Authorization", "Bearer admin-key-secret"),
                ("Authorization", "Bearer different-secret"),
            ],
        )
        entries = _audit_entries(client)

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Duplicate Authorization header is not allowed"
    )
    assert len(entries) == 1
    assert entries[0]["admin_user_id"] == "unauthenticated_admin_request"
    serialized = json.dumps(entries[0])
    assert "admin-key-secret" not in serialized
    assert "different-secret" not in serialized


def test_invalid_memory_review_filters_are_audited(
    tmp_path: Path,
) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        response = client.get(
            "/v1/admin/memory-review",
            headers={"Authorization": "Bearer admin-key-secret"},
            params={"limit": 0, "category": "not-a-category"},
        )
        entries = _audit_entries(client)

    assert response.status_code == 422
    assert len(entries) == 1
    assert entries[0]["admin_user_id"] == "admin_api_key"
    assert entries[0]["metadata_json"]["status"] == "validation_error"
    assert entries[0]["metadata_json"]["http_status"] == 422
    assert entries[0]["metadata_json"]["filters"] == {
        "limit": 0,
        "category": "not-a-category",
    }


def test_memory_review_repository_failure_is_audited(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_repository(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        raise RuntimeError("injected repository failure")

    monkeypatch.setattr(routes_admin, "_list_review_required_memories", fail_repository)
    app = create_app(_settings(tmp_path))
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get(
            "/v1/admin/memory-review",
            headers={"Authorization": "Bearer admin-key-secret"},
        )
        entries = _audit_entries(client)

    assert response.status_code == 500
    assert len(entries) == 1
    assert entries[0]["metadata_json"]["status"] == "failed"
    assert entries[0]["metadata_json"]["error_class"] == "RuntimeError"
    assert "injected repository failure" not in json.dumps(entries[0])


def test_memory_review_availability_rejection_is_audited(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def reject_unavailable_scope(*_args: Any, **_kwargs: Any) -> None:
        raise HTTPException(
            status_code=409,
            detail="sensitive maintenance detail",
        )

    monkeypatch.setattr(
        routes_admin,
        "_require_memory_scope_available",
        reject_unavailable_scope,
    )
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        response = client.get(
            "/v1/admin/memory-review",
            headers={"Authorization": "Bearer admin-key-secret"},
            params={"user_id": "usr_audit"},
        )
        entries = _audit_entries(client)

    assert response.status_code == 409
    assert len(entries) == 1
    assert entries[0]["metadata_json"]["status"] == "failed"
    assert entries[0]["metadata_json"]["error_class"] == "HTTPException"
    assert entries[0]["metadata_json"]["http_status"] == 409
    assert "sensitive maintenance detail" not in json.dumps(entries[0])


@pytest.mark.parametrize(
    ("error_type", "expected_status"),
    (
        (TranscriptRebuildInProgressError, 409),
        (TranscriptRebuildRemediationRequiredError, 503),
    ),
)
def test_memory_review_transcript_rebuild_failure_records_response_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
    expected_status: int,
) -> None:
    async def reject_unavailable_scope(*_args: Any, **_kwargs: Any) -> None:
        raise error_type("sensitive transcript maintenance detail")

    monkeypatch.setattr(
        routes_admin,
        "_require_memory_scope_available",
        reject_unavailable_scope,
    )
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        response = client.get(
            "/v1/admin/memory-review",
            headers={"Authorization": "Bearer admin-key-secret"},
            params={"user_id": "usr_audit"},
        )
        entries = _audit_entries(client)

    assert response.status_code == expected_status
    assert len(entries) == 1
    assert entries[0]["metadata_json"]["status"] == "failed"
    assert entries[0]["metadata_json"]["error_class"] == error_type.__name__
    assert entries[0]["metadata_json"]["http_status"] == expected_status
    assert "sensitive transcript maintenance detail" not in json.dumps(entries[0])


def test_memory_review_fails_closed_when_audit_storage_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_audit(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("injected audit failure")

    monkeypatch.setattr(AdminAuditRepository, "create_audit_entry", fail_audit)
    app = create_app(_settings(tmp_path))
    with TestClient(app, raise_server_exceptions=False) as client:
        _seed_sensitive_memory(client)
        response = client.get(
            "/v1/admin/memory-review",
            headers={"Authorization": "Bearer admin-key-secret"},
        )
        with _connection(client) as connection:
            cursor = client.portal.call(
                connection.execute,
                "SELECT COUNT(*) FROM admin_audit_log",
            )
            audit_count = int(client.portal.call(cursor.fetchone)[0])
            client.portal.call(cursor.close)

    assert response.status_code == 500
    assert audit_count == 0
