"""Extractor tests for language code validation and retry behavior."""

from __future__ import annotations

import pytest

from atagia.core.db_sqlite import MigrationManager
from tests.memory.test_extractor import (
    SequencedExtractionProvider,
    _build_runtime_with_provider,
    _context,
    _create_source_message,
)


async def _ensure_multilingual_schema(connection) -> None:
    cursor = await connection.execute("PRAGMA table_info(memory_objects)")
    columns = {str(row["name"]) for row in await cursor.fetchall()}
    if "language_codes_json" not in columns:
        await connection.execute("ALTER TABLE memory_objects ADD COLUMN language_codes_json TEXT")
    await connection.commit()


async def _apply_all_without_outer_begin(self, connection):
    from datetime import datetime, timezone

    await self.ensure_schema_table(connection)
    applied_versions = await self.applied_versions(connection)
    pending = [migration for migration in self.discover() if migration.version not in applied_versions]
    for migration in pending:
        timestamp = datetime.now(tz=timezone.utc).isoformat()
        try:
            if self._requires_foreign_keys_off(migration):
                await self._apply_with_foreign_keys_disabled(connection, migration, timestamp)
            else:
                await connection.executescript(migration.sql.rstrip())
                await connection.execute(
                    """
                    INSERT INTO schema_migrations(version, name, applied_at)
                    VALUES (?, ?, ?)
                    """,
                    (migration.version, migration.name, timestamp),
                )
                await connection.commit()
        except Exception:
            await connection.rollback()
            raise
    return pending


@pytest.fixture(autouse=True)
def _patch_migration_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MigrationManager, "apply_all", _apply_all_without_outer_begin)


@pytest.mark.asyncio
async def test_extraction_retries_when_language_codes_are_missing_and_echoes_validation_error() -> None:
    provider = SequencedExtractionProvider(
        [
            {
                "evidences": [
                    {
                        "canonical_text": "Tengo alergia a la penicilina.",
                        "scope": "conversation",
                        "confidence": 0.91,
                        "source_kind": "extracted",
                        "privacy_level": 0,
                        "payload": {},
                        "temporal_type": "permanent",
                        "temporal_confidence": 0.8,
                    }
                ],
                "beliefs": [],
                "contract_signals": [],
                "state_updates": [],
                "mode_guess": None,
                "nothing_durable": False,
            },
            {
                "evidences": [
                    {
                        "canonical_text": "Tengo alergia a la penicilina.",
                        "scope": "conversation",
                        "confidence": 0.91,
                        "source_kind": "extracted",
                        "privacy_level": 0,
                        "payload": {},
                        "temporal_type": "permanent",
                        "temporal_confidence": 0.8,
                        "language_codes": ["es", "ES"],
                    }
                ],
                "beliefs": [],
                "contract_signals": [],
                "state_updates": [],
                "mode_guess": None,
                "nothing_durable": False,
            },
        ],
        auto_language_codes=False,
    )
    connection, _clock, messages, _memories, extractor, sequenced_provider, resolved_policy = (
        await _build_runtime_with_provider(provider)
    )
    try:
        await _ensure_multilingual_schema(connection)
        source_message = await _create_source_message(
            messages,
            text="Tengo alergia a la penicilina.",
        )

        result = await extractor.extract(
            message_text=source_message["text"],
            role="user",
            conversation_context=_context(source_message["id"]),
            resolved_policy=resolved_policy,
        )

        assert len(sequenced_provider.requests) == 2
        retry_message = sequenced_provider.requests[1].messages[-1].content
        assert (
            "language_codes must contain at least one ISO 639-1 code for the language of canonical_text"
            in retry_message
        )
        assert "```" not in retry_message
        assert result.evidences[0].language_codes == ["es"]
    finally:
        await connection.close()
